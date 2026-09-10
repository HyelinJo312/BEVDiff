"""Sequential cache checks. Stage 1 prepares tiny configs for the training profiler.

Stage 2 checks real UNet forward/backward and Accelerator checkpoint resume.
Stage 4 forbids both raw-map loaders; run under strace to audit file opens too.
CFG/condition-encoder output comparisons are deliberately not performed.
"""

import argparse
import copy
import gc
import json
from pathlib import Path
from unittest.mock import patch

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import collate

from build_semantic_bev_cache import offline_dataset, ROOT
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from projects.bevdiffuser import data_utils
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner


def make_dataset(cfg, split='train'):
    settings = copy.deepcopy(cfg.data[split])
    settings.pop('samples_per_gpu', None)
    return build_dataset(settings, default_args=dict(pc_range=cfg.point_cloud_range,
        use_3d_bbox=cfg.use_3d_bbox, num_classes=cfg.num_classes, num_bboxes=cfg.num_bboxes))


def stage_one(cfg, args):
    dataset = offline_dataset(cfg, 'train')
    infos = dataset.data_infos[:args.samples]
    annotation = args.output / 'subset.pkl'
    mmcv.dump(dict(infos=infos, metadata=dataset.metadata), str(annotation))
    for mode in ('live', 'cached'):
        tiny = Config(copy.deepcopy(cfg._cfg_dict), filename=cfg.filename)
        for split in ('train', 'val', 'test'):
            tiny.data[split].update(ann_file=str(annotation), filter_empty_gt=False,
                use_semantic_bev_cache=mode == 'cached', semantic_bev_cache_root=str(args.cache_root))
        tiny.dump(str(args.output / (mode + '.py')))
    live = make_dataset(Config.fromfile(str(args.output / 'live.py')))
    cached = make_dataset(Config.fromfile(str(args.output / 'cached.py')))
    aligner = SegBEVAligner(**cfg.unet.parameters.seg_bev_aligner).to(args.device).eval()
    results = []
    for index in range(len(live)):
        item = live[index]
        meta = live._current_metas(item['img_metas'])
        expected, _ = aligner.project_semantics(item['seg_maps'].data.unsqueeze(0).to(args.device),
            [meta], item['depth_maps'].data.unsqueeze(0).to(args.device))
        saved = cached.semantic_bev_cache.load(meta['sample_idx'], meta)
        actual = torch.from_numpy(saved).unsqueeze(0).to(args.device)
        torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)
        assert torch.equal(expected.argmax(1), actual.argmax(1))
        assert torch.equal(expected.sum(1) == 0, actual.sum(1) == 0)
        # Exercise the actual cached training __getitem__, including temporal metadata.
        cached_item = cached[index]
        assert 'seg_maps' not in cached_item and 'depth_maps' not in cached_item
        assert torch.equal(cached_item['semantic_bev_probabilities'].data, torch.from_numpy(saved))
        results.append(dict(token=meta['sample_idx'], max_abs=float((expected - actual).abs().max()),
                            dominant_equal=True, unknown_equal=True))
        print(results[-1], flush=True)
    return dict(stage=1, device=args.device, samples=results)


def stage_two(cfg, args):
    from accelerate import Accelerator
    from model_utils import build_unet_v2

    live = make_dataset(Config.fromfile(str(args.output / 'live.py')))
    cached = make_dataset(Config.fromfile(str(args.output / 'cached.py')))
    item = live[0]
    meta = live._current_metas(item['img_metas'])
    labels = item['seg_maps'].data.unsqueeze(0).to(args.device)
    depth = item['depth_maps'].data.unsqueeze(0).to(args.device)
    probability = cached.semantic_bev_cache.load(meta['sample_idx'], meta)
    probability = torch.from_numpy(probability).unsqueeze(0).to(args.device)
    accelerator = Accelerator(cpu=args.device == 'cpu', mixed_precision='no')

    # Same serialization hooks as train_bev_diffuser_only_seg.py.
    def save_hook(models, weights, output_dir):
        for model in models:
            model.save_pretrained(str(Path(output_dir) / 'unet'))
            weights.pop()

    def load_hook(models, input_dir):
        while models:
            models.pop().from_pretrained(str(Path(input_dir) / 'unet'))

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    def prepare():
        model = build_unet_v2(cfg.unet)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return accelerator.prepare(model, optimizer)

    model, optimizer = prepare()
    keys_before = set(accelerator.unwrap_model(model).state_dict())
    x = torch.randn(1, cfg._dim_, cfg.bev_h_, cfg.bev_w_, device=args.device)
    t = torch.tensor([200], device=args.device)

    def update(model, optimizer, **condition):
        optimizer.zero_grad()
        output = model(x, t, [meta], **condition)
        assert output.shape == x.shape and torch.isfinite(output).all()
        loss = output.square().mean() + (output - x).square().mean()
        accelerator.backward(loss)
        aligner = accelerator.unwrap_model(model).seg_aligner
        assert aligner.prob_to_emb[0].weight.grad is not None
        assert torch.isfinite(aligner.prob_to_emb[0].weight.grad).all()
        assert aligner.bev_pos_embed.grad is not None
        assert any(p.grad is not None for p in aligner.seg_bev_encoder.parameters())
        optimizer.step()
        return float(loss.detach())

    live_loss = update(model, optimizer, seg_cond=labels, depth_maps=depth)
    weight = accelerator.unwrap_model(model).seg_aligner.prob_to_emb[0].weight.detach().cpu().clone()
    checkpoint = args.output / 'checkpoint-1'
    accelerator.save_state(str(checkpoint))
    model, optimizer = None, None
    accelerator.free_memory()
    gc.collect()
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    model, optimizer = prepare()
    accelerator.load_state(str(checkpoint))
    unwrapped = accelerator.unwrap_model(model)
    assert keys_before == set(unwrapped.state_dict())
    torch.testing.assert_close(unwrapped.seg_aligner.prob_to_emb[0].weight.detach().cpu(), weight, rtol=0, atol=0)
    assert {int(state['step']) for state in optimizer.state.values() if 'step' in state} == {1}
    cached_loss = update(model, optimizer, semantic_probabilities=probability)
    assert {int(state['step']) for state in optimizer.state.values() if 'step' in state} == {2}
    result = dict(stage=2, live_loss=live_loss, resumed_cached_loss=cached_loss,
                  checkpoint_keys_equal=True, restored_weight_exact=True, optimizer_step=2,
                  trainable_condition_gradients=True, output_shape=list(x.shape))
    print(result, flush=True)
    return result


def stage_four(cfg, args):
    from projects.bevdiffuser.data_utils import CustomNuScenesDiffusionDataset_seg_depth_v2 as Dataset

    settings = Config.fromfile(str(args.output / 'cached.py'))
    settings.data.train.semantic_path = '/unavailable_sam3'
    settings.data.train.depth_path = '/unavailable_metric3d'
    with patch.object(Dataset, 'load_segmaps', side_effect=AssertionError('Raw SAM3 access')) as seg, \
            patch.object(Dataset, 'load_depth_from_filenames', side_effect=AssertionError('Raw depth access')) as depth:
        dataset = make_dataset(settings)
        batch = collate([dataset[index] for index in range(min(2, len(dataset)))], samples_per_gpu=2)
        assert 'seg_maps' not in batch and 'depth_maps' not in batch
        assert 'semantic_bev_probabilities' in batch
        assert not seg.called and not depth.called
    return dict(stage=4, raw_seg_loader_calls=0, raw_depth_loader_calls=0,
                nonexistent_source_roots_succeeded=True,
                cached_batch_shape=list(torch.stack(batch['semantic_bev_probabilities'].data[0]).shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--config', default=str(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    args.output = args.output.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    result = {1: stage_one, 2: stage_two, 4: stage_four}[args.stage](cfg, args)
    (args.output / f'stage_{args.stage}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
