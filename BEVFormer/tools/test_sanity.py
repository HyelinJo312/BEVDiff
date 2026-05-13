# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
test_sanity.py  —  Stage-2 distillation sanity check

Purpose
-------
Inject Teacher's (GT-conditioned) BEV features directly into the detection head
and measure NDS/mAP. The result determines which scenario explains Stage-2 failure:

  Scenario A — Teacher BEV >> Student baseline
    → Teacher's BEV IS detection-useful.
    → Failure is in the distillation process (loss / representation mismatch).

  Scenario B — Teacher BEV ≈ Student baseline
    → Teacher's BEV is NOT detection-useful even with GT conditions.
    → Stage-1 gain comes from GT cheating at inference (privileged info),
      not from genuinely better BEV quality.

  Scenario C — Teacher BEV (without diffuser) >> Student but Teacher+Diffuser ≈ Student
    → Raw Teacher BEV is transferable but the diffuser degrades it.

Modes
-----
  student                      Standard Student forward (baseline reference)
  teacher_bev                  Teacher raw BEV (no diffuser) → Student det head
  teacher_diffuser             Teacher BEV denoised with GT seg/layout/depth → Student det head
  teacher_diffuser_layout_only Teacher BEV denoised with layout cond ONLY (no seg/depth)
  teacher_self                 Teacher BEV → Teacher's own det head (upper bound)

Usage
-----
  python -m torch.distributed.launch --nproc_per_node=N \\
      tools/test_sanity.py CONFIG STUDENT_CKPT \\
      --teacher_ckpt TEACHER_CKPT \\
      [--unet_ckpt_dir UNET_DIR]  # required for teacher_diffuser \\
      --mode {student,teacher_bev,teacher_diffuser,teacher_self} \\
      --eval bbox --launcher pytorch

Example
-------
  # Compare all three at once by running this script 3 times with different --mode.
  # Recommended order: student → teacher_bev → teacher_diffuser
"""

from __future__ import annotations

import argparse
import copy
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel, scatter
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + "/..")

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import (
    collect_results_cpu, collect_results_gpu,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BEV distillation sanity check")
    parser.add_argument("config", help="Config file (layout_tiny_seg_v4*.py)")
    parser.add_argument("student_checkpoint", help="Stage-2 Student checkpoint")
    parser.add_argument("--teacher_ckpt", required=True,
                        help="Stage-1 Teacher checkpoint (bevformer_tiny epoch24)")
    parser.add_argument("--unet_ckpt_dir", default=None,
                        help="Stage-1 UNet checkpoint dir (required for teacher_diffuser)")
    parser.add_argument(
        "--mode",
        choices=["student", "teacher_bev", "teacher_diffuser",
                 "teacher_diffuser_layout_only", "teacher_self"],
        default="teacher_diffuser",
        help=(
            "student: Student baseline | "
            "teacher_bev: Teacher raw BEV → Student det head | "
            "teacher_diffuser: Teacher+Diffuser GT BEV → Student det head | "
            "teacher_diffuser_layout_only: layout cond only (no seg/depth) | "
            "teacher_self: Teacher BEV → Teacher det head (upper bound)"
        ),
    )
    parser.add_argument("--eval", type=str, nargs="+", default=["bbox"])
    parser.add_argument("--out", default=None, help="Output pkl file")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# ---------------------------------------------------------------------------
# GT condition loading
# ---------------------------------------------------------------------------

def build_token_index(dataset) -> dict:
    """Build sample_token → data_infos index map."""
    return {info["token"]: i for i, info in enumerate(dataset.data_infos)}


def load_gt_conditions(dataset, sample_token: str, img_metas: dict,
                       device: torch.device, token_to_idx: dict,
                       layout_only: bool = False):
    """Load GT conditions for one sample.

    Delegates to dataset methods from data_utils.py — load_segmaps,
    get_ann_info, get_layout_info, load_depth_from_filenames — so the
    conditioning logic is identical to Stage-1 training.

    Parameters
    ----------
    layout_only : bool
        If True, skip loading seg_maps and depth_maps entirely.
        Only the layout condition (cond) is returned.  Used by the
        ``teacher_diffuser_layout_only`` ablation baseline.

    Returns
    -------
    seg_maps   : LongTensor (1, V, H, W) or None
    cond       : dict with 'obj_class', 'obj_bbox', 'is_valid_obj' on *device*
    depth_maps : FloatTensor (1, V, H, W) or None
    """
    from mmcv.parallel import DataContainer as DC

    idx = token_to_idx.get(sample_token)
    if idx is None:
        return None, {}, None

    filenames = img_metas["filename"]

    # --- seg_maps (dataset.load_segmaps) ---
    # [layout_only] skip seg_maps entirely for layout-only ablation
    seg_maps = None
    seg_class_valid = []
    if not layout_only:
        if getattr(dataset, "use_semantics", False) and getattr(dataset, "semantic_path", None):
            seg_maps = dataset.load_segmaps(filenames, img_metas, dataset.semantic_path)
            seg_maps = seg_maps.unsqueeze(0).to(device)  # (1, V, H, W)
            seg_class_valid = seg_maps[0].unique().tolist()

    # --- layout (dataset.get_ann_info → get_layout_info) ---
    # Wrap in DC so get_layout_info's isinstance/while-unwrap logic passes.
    annos = dataset.get_ann_info(idx)
    data_for_layout = {}
    if annos.get("gt_labels_3d") is not None:
        data_for_layout["gt_labels_3d"] = DC(
            torch.from_numpy(annos["gt_labels_3d"]).long())
    if annos.get("gt_bboxes_3d") is not None:
        data_for_layout["gt_bboxes_3d"] = DC(annos["gt_bboxes_3d"])

    # [layout_only] CustomNuScenesDiffusionDataset_layout.get_layout_info(data)
    # takes only 1 arg, while the _seg variant takes 2 (data, seg_class_valid).
    import inspect
    sig = inspect.signature(dataset.get_layout_info)
    if len(sig.parameters) >= 2:
        layout = dataset.get_layout_info(data_for_layout, seg_class_valid)
    else:
        layout = dataset.get_layout_info(data_for_layout)
    cond = {
        "obj_class":    layout["layout_obj_classes"].data.unsqueeze(0).to(device),
        "obj_bbox":     layout["layout_obj_bboxes"].data.unsqueeze(0).to(device),
        "is_valid_obj": layout["layout_obj_is_valid"].data.unsqueeze(0).to(device),
    }

    # --- depth_maps (dataset.load_depth_from_filenames) ---
    # [layout_only] skip depth_maps entirely for layout-only ablation
    depth_maps = None
    if not layout_only:
        if getattr(dataset, "use_depth", False) and getattr(dataset, "depth_path", None):
            depth_maps = dataset.load_depth_from_filenames(filenames)
            depth_maps = depth_maps.unsqueeze(0).to(device)  # (1, V, H, W)

    return seg_maps, cond, depth_maps


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_test(model, teacher, bev_diffuser, data_loader, dataset,
                bev_h: int, bev_w: int, mode: str,
                tmpdir, gpu_collect: bool):
    """Inference loop with optional Teacher BEV injection.

    Temporal tracking for Teacher follows the same logic as
    BEVFormer.forward_test (scene-aware prev_bev + ego-motion delta).
    The custom DistributedSampler assigns CONTIGUOUS index chunks to each GPU,
    so temporal consistency is preserved within each GPU's portion.
    """
    model.eval()
    if teacher is not None:
        teacher.eval()
    if bev_diffuser is not None:
        bev_diffuser.eval()

    rank, world_size = get_dist_info()
    bbox_results = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)

    # Pre-build GT condition index (only needed for teacher modes)
    token_to_idx = build_token_index(dataset) if mode != "student" else {}

    # Teacher temporal state — mimics BEVFormer.prev_frame_info
    t_frame = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}

    for i, data in enumerate(data_loader):
        # --- Student baseline: reuse existing test infrastructure ---
        if mode == "student":
            result = model(return_loss=False, rescale=True, **data)
            batch_size = len(result)
            bbox_results.extend(result)
            if rank == 0:
                for _ in range(batch_size * world_size):
                    prog_bar.update()
            continue

        # --- Teacher modes: scatter data manually, call model components ---
        # scatter() moves DataContainers to GPU and unwraps them.
        gpu_data = scatter(data, [torch.cuda.current_device()])[0]
        # After scatter:
        #   gpu_data['img'][0]        : (B, N_cam, C, H, W) tensor on CUDA
        #   gpu_data['img_metas'][0]  : [dict, ...] list of B dicts
        img       = gpu_data["img"][0]           # (B, N, C, H, W)
        img_metas = gpu_data["img_metas"][0]     # list[dict], len=B

        B = img.shape[0]
        assert B == 1, (
            "Teacher modes require samples_per_gpu=1 for correct temporal tracking.")
        cur_meta = img_metas[0]
        scene_token  = cur_meta["scene_token"]
        sample_token = cur_meta["sample_idx"]  # == nuScenes sample token

        # Teacher temporal: reset on new scene
        if scene_token != t_frame["scene_token"]:
            t_frame["prev_bev"] = None
        t_frame["scene_token"] = scene_token

        # Ego-motion delta (same as forward_test)
        tmp_pos   = copy.deepcopy(cur_meta["can_bus"][:3])
        tmp_angle = copy.deepcopy(cur_meta["can_bus"][-1])
        if t_frame["prev_bev"] is not None:
            cur_meta["can_bus"][:3] -= t_frame["prev_pos"]
            cur_meta["can_bus"][-1] -= t_frame["prev_angle"]
        else:
            cur_meta["can_bus"][:3] = 0
            cur_meta["can_bus"][-1] = 0

        # Teacher: extract img features → build BEV
        # Clone img to prevent extract_img_feat's in-place squeeze_() from
        # corrupting the (1,N,C,H,W) shape needed by the Student call below.
        t_img_feats = teacher.extract_feat(img=img.clone(), img_metas=[cur_meta])
        teacher_bev = teacher.pts_bbox_head(
            t_img_feats, [cur_meta], t_frame["prev_bev"], only_bev=True
        )  # (1, H*W, C)

        # Update Teacher temporal state
        t_frame["prev_bev"]   = teacher_bev.detach()
        t_frame["prev_pos"]   = tmp_pos
        t_frame["prev_angle"] = tmp_angle

        C   = teacher_bev.shape[-1]
        H, W = bev_h, bev_w

        # --- mode: teacher_self  (Teacher BEV → Teacher det head) ---
        if mode == "teacher_self":
            outs = teacher.pts_bbox_head(
                t_img_feats, [cur_meta], prev_bev=None, given_bev=teacher_bev)
            bbox_list = teacher.pts_bbox_head.get_bboxes(
                outs, [cur_meta], rescale=True)
            bbox_pts = [bbox3d2result(b, s, l) for b, s, l in bbox_list]
            result = [{"pts_bbox": r} for r in bbox_pts]
            bbox_results.extend(result)
            if rank == 0:
                for _ in range(B * world_size):
                    prog_bar.update()
            continue

        # --- mode: teacher_diffuser  (full guidance: seg + depth + layout) ---
        if mode == "teacher_diffuser":
            assert bev_diffuser is not None, "--unet_ckpt_dir required for teacher_diffuser"
            seg_maps, cond, depth_maps = load_gt_conditions(
                dataset, sample_token, cur_meta,
                device=img.device, token_to_idx=token_to_idx)

            # Reshape: (1, H*W, C) → (1, C, H, W) for diffuser
            bev_2d = teacher_bev.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous().float()

            if seg_maps is not None and cond:
                denoised = bev_diffuser(bev_2d, [cur_meta], cond, seg_maps, depth_maps)
            else:
                # No seg available — fall back to raw Teacher BEV (unconditioned)
                print(f"[warn] sample {sample_token}: no seg_maps, skipping diffuser")
                denoised = bev_2d

            given_bev = denoised.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # --- mode: teacher_diffuser_layout_only  (layout cond ONLY, no seg/depth) ---
        elif mode == "teacher_diffuser_layout_only":
            assert bev_diffuser is not None, "--unet_ckpt_dir required for teacher_diffuser_layout_only"
            # Load only the layout condition; seg_maps and depth_maps are skipped.
            _, cond, _ = load_gt_conditions(
                dataset, sample_token, cur_meta,
                device=img.device, token_to_idx=token_to_idx,
                layout_only=True)

            # Reshape: (1, H*W, C) → (1, C, H, W) for diffuser
            bev_2d = teacher_bev.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous().float()

            if cond:
                # Layout-only BEVDiffuser: forward(x, condition)
                denoised = bev_diffuser(bev_2d, cond)
            else:
                print(f"[warn] sample {sample_token}: no cond, skipping diffuser")
                denoised = bev_2d

            given_bev = denoised.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            # --- mode: teacher_bev  (raw Teacher BEV, no diffuser) ---
            given_bev = teacher_bev  # (1, H*W, C)

        # Student img features (for object query cross-attention in decoder)
        s_img_feats = model.module.extract_feat(img=img.clone(), img_metas=[cur_meta])

        # Inject Teacher BEV into Student detection head
        outs = model.module.pts_bbox_head(
            s_img_feats, [cur_meta], prev_bev=None, given_bev=given_bev)
        bbox_list = model.module.pts_bbox_head.get_bboxes(
            outs, [cur_meta], rescale=True)
        bbox_pts = [bbox3d2result(b, s, l) for b, s, l in bbox_list]
        result = [{"pts_bbox": r} for r in bbox_pts]
        bbox_results.extend(result)

        if rank == 0:
            for _ in range(B * world_size):
                prog_bar.update()

    if gpu_collect:
        return collect_results_gpu(bbox_results, len(dataset))
    else:
        return collect_results_cpu(bbox_results, len(dataset), tmpdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Load plugin
    if getattr(cfg, "plugin", False):
        import importlib
        if hasattr(cfg, "plugin_dir"):
            _parts = os.path.dirname(cfg.plugin_dir).split("/")
            _mod   = _parts[0]
            for p in _parts[1:]:
                _mod += "." + p
            importlib.import_module(_mod)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("close_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None

    # Build test dataset
    samples_per_gpu = 1  # always 1 for teacher modes (temporal consistency)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        _ = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset_default_args = {
        "pc_range":    cfg.point_cloud_range,
        "use_3d_bbox": cfg.use_3d_bbox,
        "num_classes": cfg.num_classes,
        "num_bboxes":  cfg.num_bboxes,
    }
    dataset = build_dataset(cfg.data.test, default_args=dataset_default_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # Build Student model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.student_checkpoint, map_location="cpu")
    model.CLASSES = dataset.CLASSES

    if not distributed:
        raise ValueError("Use --launcher pytorch. Single-GPU mode not supported.")
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )

    # Build Teacher (same architecture, Stage-1 checkpoint)
    teacher = None
    if args.mode != "student":
        teacher_cfg = copy.deepcopy(cfg)
        teacher_cfg.model.train_cfg = None
        teacher = build_model(teacher_cfg.model, test_cfg=teacher_cfg.get("test_cfg"))
        if fp16_cfg is not None:
            wrap_fp16_model(teacher)
        load_checkpoint(teacher, args.teacher_ckpt, map_location="cpu")
        teacher = teacher.cuda()
        teacher.requires_grad_(False)
        teacher.eval()

    # Build BEVDiffuser
    bev_diffuser = None
    if args.mode == "teacher_diffuser":
        # Full-guidance diffuser (seg + depth + layout)
        if args.unet_ckpt_dir is None:
            raise ValueError("--unet_ckpt_dir is required for mode=teacher_diffuser")
        from bevdiffuser_seg import BEVDiffuser  # noqa: local import after sys.path setup
        diffuser_cfg = copy.deepcopy(dict(cfg.bev_diffuser_cfg))
        diffuser_cfg["unet_checkpoint_dir"] = args.unet_ckpt_dir
        bev_diffuser = BEVDiffuser(**diffuser_cfg)
        bev_diffuser = bev_diffuser.cuda()
        bev_diffuser.requires_grad_(False)
        bev_diffuser.eval()
    elif args.mode == "teacher_diffuser_layout_only":
        # [layout_only] Layout-only diffuser — no seg/depth encoder in UNet.
        # Uses bevdiffuser.BEVDiffuser whose forward is: forward(x, condition).
        if args.unet_ckpt_dir is None:
            raise ValueError("--unet_ckpt_dir is required for mode=teacher_diffuser_layout_only")
        from bevdiffuser import BEVDiffuser  # noqa: layout-only variant
        diffuser_cfg = copy.deepcopy(dict(cfg.bev_diffuser_cfg))
        diffuser_cfg["unet_checkpoint_dir"] = args.unet_ckpt_dir
        bev_diffuser = BEVDiffuser(**diffuser_cfg)
        bev_diffuser = bev_diffuser.cuda()
        bev_diffuser.requires_grad_(False)
        bev_diffuser.eval()

    bev_h = getattr(cfg, "bev_h_", 50)
    bev_w = getattr(cfg, "bev_w_", 50)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Sanity check mode : {args.mode}")
        print(f"  Student ckpt      : {args.student_checkpoint}")
        print(f"  Teacher ckpt      : {args.teacher_ckpt}")
        if bev_diffuser is not None:
            print(f"  UNet ckpt dir     : {args.unet_ckpt_dir}")
        print(f"  Dataset size      : {len(dataset)}")
        print(f"{'='*60}\n")

    # Run inference
    outputs = sanity_test(
        model, teacher, bev_diffuser, data_loader, dataset,
        bev_h, bev_w, args.mode, args.tmpdir, args.gpu_collect,
    )

    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)

        ts          = time.ctime().replace(" ", "_").replace(":", "_")
        cfg_name    = args.config.split("/")[-1].split(".")[0]
        json_prefix = osp.join("test_sanity", f"{cfg_name}_{args.mode}", ts)
        mmcv.mkdir_or_exist(osp.dirname(json_prefix))

        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, jsonfile_prefix=json_prefix))
            print(f"\n[{args.mode}] Evaluation results:")
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
