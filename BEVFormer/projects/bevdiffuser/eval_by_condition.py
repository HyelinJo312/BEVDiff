"""
Per-condition nuScenes detection evaluation (sunny / rainy / day / night).

Predictions are condition-independent, so this script does NOT re-run inference.
It reuses an existing `results_nusc.json` (produced by a normal test run) and
re-evaluates it on subsets of samples grouped by the nuScenes scene description.

Scene descriptions are free text; we use the standard keyword convention:
    rainy : 'rain'  in description
    sunny : 'rain'  NOT in description   (i.e. not raining / clear)
    night : 'night' in description
    day   : 'night' NOT in description
The 'sunny/rainy' and 'day/night' splits are two independent partitions of the
full val set, so a sample appears in exactly one of {sunny,rainy} and one of
{day,night}.

Usage:
    python eval_by_condition.py \
        --bev_config ../configs/diff_bevformer/layout_tiny_seg_v4_2.py \
        --result_path ../../test/<cfg>/<run>/<ckpt>/5_5_5/results_nusc.json
"""

import argparse
import os
import sys

# Make the BEVFormer root importable. The mmdet3d plugin's datasets/__init__.py
# imports projects.bevdiffuser.data_utils, which registers the custom diffusion
# dataset types — so we must NOT import data_utils directly here (double-register).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

from mmcv import Config
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes

from projects.mmdet3d_plugin.datasets.nuscnes_eval import (
    NuScenesEval_custom,
    evaluate_by_scene_condition,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-condition nuScenes detection evaluation."
    )
    parser.add_argument("--bev_config", required=True, help="test config file path")
    parser.add_argument(
        "--result_path",
        required=True,
        help="path to results_nusc.json produced by a previous test run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="folder for eval scratch output (default: alongside result_path)",
    )
    return parser.parse_args()


def load_plugin(cfg):
    """Import plugin modules so build_dataset can resolve custom dataset types."""
    if cfg.get("plugin", False) and cfg.get("plugin_dir", None):
        import importlib

        module_path = ".".join(os.path.dirname(cfg.plugin_dir).split("/"))
        importlib.import_module(module_path)


def build_test_dataset(cfg, data_root_abs):
    cfg.data.test.test_mode = True
    cfg.data.test.load_annos = True
    # Configs disagree on the data_root prefix ('data/nuscenes/' vs
    # 'BEVFormer/data/nuscenes/'); force absolute paths so eval works regardless
    # of which config / CWD is used.
    cfg.data.test.data_root = data_root_abs
    cfg.data.test.ann_file = os.path.join(
        data_root_abs, os.path.basename(cfg.data.test.ann_file)
    )
    return build_dataset(
        cfg.data.test,
        default_args={
            "pc_range": cfg.point_cloud_range,
            "use_3d_bbox": cfg.use_3d_bbox,
            "num_classes": cfg.num_classes,
            "num_bboxes": cfg.num_bboxes,
        },
    )


def main():
    args = parse_args()

    # Resolve user paths before chdir, then run from the BEVFormer root.
    args.bev_config = os.path.abspath(args.bev_config)
    args.result_path = os.path.abspath(args.result_path)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    bevformer_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(bevformer_root)
    # The real nuScenes lives at <BEVFormer>/data/nuscenes regardless of the
    # data_root string in the config.
    data_root_abs = os.path.join(bevformer_root, "data", "nuscenes") + "/"

    cfg = Config.fromfile(args.bev_config)
    load_plugin(cfg)

    dataset = build_test_dataset(cfg, data_root_abs)

    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=True)
    eval_set_map = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}

    output_dir = args.output_dir or os.path.join(
        *os.path.split(args.result_path)[:-1], "by_condition"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions + GT once; all_gt / all_preds hold the full filtered set.
    nusc_eval = NuScenesEval_custom(
        nusc,
        config=dataset.eval_detection_configs,
        result_path=args.result_path,
        eval_set=eval_set_map[dataset.version],
        output_dir=output_dir,
        verbose=False,
        overlap_test=False,
        data_infos=dataset.data_infos,
    )

    evaluate_by_scene_condition(nusc_eval, nusc, result_path=args.result_path)


if __name__ == "__main__":
    main()
