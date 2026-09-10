"""Compare v5 against a saved pre-optimization source on real and synthetic maps."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from mmcv import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'projects.bevdiffuser.layout_diffusion._reference_v5', args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cfg = Config.fromfile(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py')
    old = reference.SegBEVAligner(**cfg.unet.parameters.seg_bev_aligner).to(args.device).eval()
    new = SegBEVAligner(**cfg.unet.parameters.seg_bev_aligner).to(args.device).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    device = torch.device(args.device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def compare(seg, meta, depth):
        a = old.project_semantics(seg, meta, depth)
        b = new.project_semantics(seg, meta, depth)
        torch.testing.assert_close(a[0], b[0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(a[1], b[1], rtol=2e-5, atol=2e-3)
        assert torch.equal(a[1] > 0, b[1] > 0)
        return dict(probability_max_abs=float((a[0] - b[0]).abs().max()),
                    support_max_abs=float((a[1] - b[1]).abs().max()),
                    dominant_changed=int((a[0].argmax(1) != b[0].argmax(1)).sum()))

    torch.manual_seed(0)
    synthetic = []
    for stride in (1, 2):
        for mode in ('nearest', 'bilinear'):
            for depth_type in ('camera_z', 'ray_distance'):
                old.pixel_stride = new.pixel_stride = stride
                old.splat_mode = new.splat_mode = mode
                old.depth_type = new.depth_type = depth_type
                seg = torch.randint(-1, 17, (2, 2, 16, 24), device=device).float()
                seg[0, 0, 0, :3] = torch.tensor([1.5, float('nan'), 16], device=device)
                depth = torch.rand_like(seg) * 180
                depth[0, 0, 1, :3] = torch.tensor([0, float('nan'), float('inf')], device=device)
                meta = [dict(lidar2img=[np.diag([10., 10., 1., 1.])] * 2)] * 2
                synthetic.append(compare(seg, meta, depth))
                synthetic.append(compare(torch.zeros_like(seg), meta, depth))
    old.pixel_stride = new.pixel_stride = cfg.unet.parameters.seg_bev_aligner.pixel_stride
    old.splat_mode = new.splat_mode = cfg.unet.parameters.seg_bev_aligner.splat_mode
    old.depth_type = new.depth_type = cfg.unet.parameters.seg_bev_aligner.depth_type

    settings = cfg.data.test.copy()
    settings.update(test_mode=True, load_annos=True)
    dataset = build_dataset(settings, default_args=dict(pc_range=cfg.point_cloud_range,
        use_3d_bbox=cfg.use_3d_bbox, num_classes=cfg.num_classes, num_bboxes=cfg.num_bboxes))
    cv2.setNumThreads(1)
    samples, real = [], []
    for index in (0, 39, 79, 119, 158, 197):
        start = time.perf_counter()
        item = dataset[index]
        load_s = time.perf_counter() - start
        meta = dataset._current_metas(item['img_metas'])
        seg = item['seg_maps'].data.to(device).unsqueeze(0)
        depth = item['depth_maps'].data.to(device).unsqueeze(0)
        result = compare(seg, [meta], depth)
        result.update(index=index, sample_token=meta['sample_idx'], load_s=load_s)
        real.append(result)
        samples.append((seg, meta, depth))
        print(result, flush=True)
    seg = torch.cat([s[0] for s in samples[:2]])
    meta = [s[1] for s in samples[:2]]
    depth = torch.cat([s[2] for s in samples[:2]])
    batch_comparison = compare(seg, meta, depth)
    timings = {}
    for condition, labels in [('observed', seg), ('cfg_dropout', torch.zeros_like(seg))]:
        timings[condition] = {'before': [], 'after': []}
        for model in (old, new):
            model.project_semantics(labels, meta, depth)
        for repeat in range(args.repeats):
            order = [('before', old), ('after', new)]
            for name, model in order[::1 if repeat % 2 == 0 else -1]:
                sync()
                start = time.perf_counter()
                model.project_semantics(labels, meta, depth)
                sync()
                timings[condition][name].append(time.perf_counter() - start)
    operators = {}
    for name, model in [('before', old), ('after', new)]:
        with torch.autograd.profiler.profile(use_cuda=device.type == 'cuda') as profile:
            model.project_semantics(seg, meta, depth)
        operators[name] = {entry.key: entry.count for entry in profile.key_averages()
                           if 'nonzero' in entry.key or 'synchronize' in entry.key.lower()
                           or entry.key == 'aten::scatter_add_'}
    preprocessing = {}
    dataset.opencv_num_threads = 32
    threaded = dataset[0]
    dataset.opencv_num_threads = 1
    serial = dataset[0]
    for key in ('seg_maps', 'depth_maps'):
        assert torch.equal(threaded[key].data, serial[key].data)
        preprocessing[key + '_exact'] = True
    assert torch.equal(threaded['img'][0].data, serial['img'][0].data)
    preprocessing['image_exact'] = True
    preprocessing['effective_opencv_threads'] = cv2.getNumThreads()
    report = dict(device=str(device), synthetic_cases=len(synthetic), synthetic=synthetic,
        real=real, batch_comparison=batch_comparison, timings_s=timings, operators=operators,
        preprocessing=preprocessing,
        tolerance=dict(probability_rtol=2e-5, probability_atol=2e-6,
                       support_rtol=2e-5, support_atol=2e-3))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(timings={key:{name:float(np.median(values)) for name,values in value.items()}
                                  for key,value in timings.items()}, operators=operators)), flush=True)


if __name__ == '__main__':
    main()
