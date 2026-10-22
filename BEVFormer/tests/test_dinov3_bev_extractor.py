"""Offline geometry, frozen target and Stage 1 integration checks."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / 'projects' / 'bevdiffuser'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))
from dinov3_bev_extractor import FrozenDINOv3Encoder, MetricDepthFeatureProjector, DINOv3BEVExtractor
from train_bev_diffuser_seg_dino import unpack_batch, diffusion_target, predict_clean, drop_condition, load_unet_strict, attach_cls_condition


def stage_one_config(use_global_cls=False):
    """Derive both test variants from the standalone config kept in the repo."""
    from mmcv import Config
    cfg = Config.fromfile(str(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_dino.py'))
    if not use_global_cls:
        cfg.unet.type = 'layout_diffusion.seg_diffusion_unet_v5.DiffusionUNetModel'
        for key in ('use_global_cls', 'cls_channels', 'cls_camera_order'):
            cfg.unet.parameters.pop(key)
        cfg.unet.parameters.use_global_seg = True
    return cfg


def metadata(height, width, views=1):
    return dict(filename=['image'] * views, sample_idx='sample',
                img_shape=[(height, width, 3)] * views, ori_shape=[(height, width, 3)] * views,
                pad_shape=[(height, width, 3)] * views,
                lidar2img=np.repeat(np.eye(4)[None], views, axis=0),
                metric3d_lidar2img_raw=np.repeat(np.eye(4)[None], views, axis=0),
                img_norm_cfg=dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True))


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384, patch_size=16, num_register_tokens=4)
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.inputs = []

    def forward(self, pixel_values, return_dict=True):
        self.inputs.append(pixel_values.clone())
        b, _, h, w = pixel_values.shape
        tokens = self.anchor * torch.ones(b, 5 + h // 16 * (w // 16), 384, device=pixel_values.device)
        tokens[:, 0] = 7
        tokens[:, 1:5] = 99  # Must never leak register tokens into the patch grid.
        return SimpleNamespace(last_hidden_state=tokens)


class EncoderTest(unittest.TestCase):
    def test_frozen_cls_registers_non_square_padding_and_chunking(self):
        model = FakeDINO()
        encoder = FrozenDINOv3Encoder(model=model, view_batch_size=2)
        extractor = DINOv3BEVExtractor(encoder=encoder)
        nn.Sequential(extractor).train()
        self.assertFalse(model.training)
        self.assertFalse(extractor.training)
        images = torch.ones(2, 3, 3, 17, 32, requires_grad=True) * 127
        result = encoder(images, [metadata(17, 32, 3)] * 2)
        self.assertEqual(result['patch_features'].shape, (2, 3, 384, 2, 2))
        self.assertEqual(result['cls_tokens'].shape, (2, 3, 384))
        self.assertEqual(result['feature_image_shape'], (32, 32))
        self.assertEqual(len(model.inputs), 3)
        self.assertFalse(result['patch_features'].requires_grad)
        self.assertFalse(result['cls_tokens'].requires_grad)
        self.assertTrue(all(not p.requires_grad for p in extractor.parameters()))
        torch.testing.assert_close(result['patch_features'], torch.ones_like(result['patch_features']))
        torch.testing.assert_close(result['cls_tokens'], torch.full_like(result['cls_tokens'], 7))

    def test_huggingface_local_checkpoint_and_real_dinov3_tokens(self):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if 'dinov3_vit' not in CONFIG_MAPPING:
            self.skipTest('Installed Transformers has no DINOv3; run with 4.56.2')
        from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
        config = DINOv3ViTConfig(hidden_size=384, intermediate_size=768, num_hidden_layers=1,
                                num_attention_heads=6, num_register_tokens=4, patch_size=16)
        with tempfile.TemporaryDirectory() as tmp:
            DINOv3ViTModel(config).save_pretrained(tmp)
            DINOv3ViTImageProcessorFast().save_pretrained(tmp)
            encoder = FrozenDINOv3Encoder(model_name=tmp, local_files_only=True)
            images = torch.randn(1, 2, 3, 17, 48) * 30 + 128
            metas = [metadata(17, 48, 2)]
            result = encoder(images, metas)
            self.assertEqual(result['patch_features'].shape, (1, 2, 384, 2, 3))
            self.assertEqual(result['cls_tokens'].shape, (1, 2, 384))
            self.assertFalse(result['patch_features'].requires_grad)
            self.assertTrue(torch.isfinite(result['patch_features']).all())
            torch.testing.assert_close(encoder(images, metas)['cls_tokens'], result['cls_tokens'])

    def test_undo_bgr_normalization_and_preserve_padding(self):
        encoder = FrozenDINOv3Encoder(model=FakeDINO())
        meta = metadata(16, 32)
        meta['ori_shape'] = [(15, 31, 3)]
        meta['img_norm_cfg'] = dict(mean=[10, 20, 30], std=[2, 3, 4], to_rgb=False)
        bgr = torch.tensor([25., 100., 200.]).view(1, 1, 3, 1, 1).expand(1, 1, 3, 16, 32)
        images = (bgr - torch.tensor([10, 20, 30]).view(1, 1, 3, 1, 1)) / torch.tensor([2, 3, 4]).view(1, 1, 3, 1, 1)
        pixels = encoder.preprocess(images, [meta])
        expected = (torch.tensor([200., 100., 25.]) / 255 - encoder.image_mean.flatten()) / encoder.image_std.flatten()
        torch.testing.assert_close(pixels[0, :, 0, 0], expected)
        self.assertEqual(pixels[0, :, 15].abs().sum(), 0)
        self.assertEqual(pixels[0, :, :, 31].abs().sum(), 0)


class ProjectionTest(unittest.TestCase):
    def projector(self, **kwargs):
        options = dict(bev_h=4, bev_w=4, pc_range=(-.5, -.5, .5, 3.5, 3.5, 1.5),
                       depth_range=(.1, 10), splat_mode='nearest')
        options.update(kwargs)
        return MetricDepthFeatureProjector(**options)

    def test_identity_coordinates_signed_channels_and_unknown_cells(self):
        features = torch.tensor([[[[[1., 2.], [3., 4.]], [[-1., -2.], [-3., -4.]]]]])
        result = self.projector()(features, torch.ones(1, 1, 2, 2), [metadata(2, 2)], (2, 2))
        expected = torch.zeros(1, 2, 4, 4)
        expected[:, :, :2, :2] = features[:, 0]
        torch.testing.assert_close(result['bev_features'], expected)
        self.assertEqual(result['observed_mask'].sum(), 4)
        self.assertEqual(result['support'].sum(), 4)

    def test_calibration_translation_and_invalid_depth(self):
        meta = metadata(2, 2)
        meta['lidar2img'][0, 0, 3] = -1  # Camera origin is at lidar x=+1.
        depths = torch.tensor([[[[1., float('nan')], [0., 20.]]]])
        result = self.projector()(torch.ones(1, 1, 1, 1, 1), depths, [meta], (2, 2))
        self.assertEqual(result['observed_mask'].sum(), 1)
        self.assertEqual(result['bev_features'][0, 0, 0, 1], 1)

    def test_augmented_intrinsics_and_ray_distance(self):
        meta = metadata(1, 3)
        meta['lidar2img'][0, 0, 0] = 2
        depth = torch.tensor([[[[0., 0., 1.]]]])
        p = self.projector(bev_h=4, bev_w=4, pc_range=(0, -.5, .5, 2, 1.5, 1.5))
        result = p(torch.ones(1, 1, 1, 1, 1), depth, [meta], (1, 3))
        self.assertEqual(result['bev_features'][0, 0, 1, 2], 1)  # x=2/2=1
        p.depth_type = 'ray_distance'
        result = p(torch.ones(1, 1, 1, 1, 1), depth, [meta], (1, 3))
        self.assertEqual(result['bev_features'][0, 0, 1, 1], 1)  # x=1/sqrt(2)

    def test_bilinear_mass_and_signed_vector_are_not_attenuated(self):
        meta = metadata(1, 1)
        meta['lidar2img'][0, :2, 3] = -.25
        result = self.projector(splat_mode='bilinear')(
            torch.tensor([[[[[-2.]]]]]), torch.ones(1, 1, 1, 1), [meta], (1, 1))
        torch.testing.assert_close(result['support'][0, 0, :2, :2], torch.tensor([[.5625, .1875], [.1875, .0625]]))
        torch.testing.assert_close(result['bev_features'][0, 0, :2, :2], torch.full((2, 2), -2.))

    def test_camera_mean_is_independent_of_depth_sampling_density(self):
        features = torch.stack([torch.ones(1, 2, 2), torch.full((1, 2, 2), 3.)])[None]
        depth = torch.ones(1, 2, 2, 2)
        depth[:, 1] = 0
        depth[:, 1, 0, 0] = 1
        result = self.projector(bev_h=1, bev_w=1)(features, depth, [metadata(2, 2, 2)], (2, 2))
        self.assertEqual(result['bev_features'].item(), 2)
        self.assertEqual(result['support'].item(), 5)

    def test_chunked_splat_matches_full_and_empty_input(self):
        torch.manual_seed(0)
        features = torch.randn(1, 2, 3, 2, 2)
        depth = torch.ones(1, 2, 4, 4)
        meta = [metadata(4, 4, 2)]
        full = self.projector(splat_mode='bilinear')(features, depth, meta, (4, 4))
        chunked = self.projector(splat_mode='bilinear', point_chunk_size=1)(features, depth, meta, (4, 4))
        for key in full:
            torch.testing.assert_close(full[key], chunked[key])
        empty = self.projector()(features, depth * 0, meta, (4, 4))
        self.assertEqual(empty['bev_features'].abs().sum(), 0)
        self.assertEqual(empty['support'].sum(), 0)
        self.assertFalse(empty['observed_mask'].any())


class StageOneIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import projects.mmdet3d_plugin  # noqa: F401
        from projects.bevdiffuser.data_utils import CustomNuScenesDiffusionDataset_seg_depth_v2, CustomNuScenesDataset
        cls.Dataset, cls.Base = CustomNuScenesDiffusionDataset_seg_depth_v2, CustomNuScenesDataset

    def test_cache_depth_loading_preserves_raw_metadata_and_unet_condition(self):
        from mmcv.parallel import DataContainer as DC, collate
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CustomCollect3D
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'sample.npy', np.ones((17, 50, 50), dtype=np.float32))
            (root / 'samples').mkdir()
            np.save(root / 'samples' / 'image.npy', np.ones((2, 2), dtype=np.float32))
            def init(base, **kwargs):
                base.pipeline = SimpleNamespace(transforms=[CustomCollect3D(keys=['img'])])
            with patch.object(self.Base, '__init__', init):
                dataset = self.Dataset(pc_range=[-.5, -.5, .5, 3.5, 3.5, 1.5],
                                       use_semantics=True, use_depth=True, depth_path=tmp,
                                       depth_raw_shape=(2, 2), depth_range=(.1, 10),
                                       use_semantic_bev_cache=True, semantic_bev_cache_root=tmp,
                                       semantic_bev_cache_shape=(17, 50, 50), load_depth_in_cache=True)
            self.assertIn('metric3d_lidar2img_raw', dataset.pipeline.transforms[0].meta_keys)
            meta = metadata(2, 2)
            raw = dict(img=DC(torch.ones(1, 1, 3, 2, 2), stack=True), img_metas=DC({0: meta}, cpu_only=True))
            with patch.object(self.Base, '__getitem__', return_value=raw), \
                    patch.object(dataset, 'load_segmaps', side_effect=AssertionError('Unexpected SAM access')):
                item = dataset[0]
            batch = collate([item, item], samples_per_gpu=2)
            images, metas, depth, condition = unpack_batch(batch, 'cpu')
            self.assertEqual(images.shape, (2, 1, 3, 2, 2))
            self.assertEqual(depth.shape, (2, 1, 2, 2))
            self.assertEqual(condition['semantic_prob'].shape, (2, 17, 50, 50))
            self.assertNotIn('depth_maps', condition)
            self.assertIsNone(condition['seg_cond'])
            # One test augmentation uses a different DataContainer layout.
            val_item = dict(img=[DC(torch.ones(1, 3, 2, 2), stack=True)],
                            img_metas=[DC(meta, cpu_only=True)], depth_maps=item['depth_maps'],
                            semantic_bev_probabilities=item['semantic_bev_probabilities'])
            self.assertEqual(unpack_batch(collate([val_item], samples_per_gpu=1), 'cpu')[0].shape, (1, 1, 3, 2, 2))

    def test_config_camera_order_aligns_images_and_calibration(self):
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CustomCollect3D
        order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        def init(base, **kwargs):
            base.pipeline = SimpleNamespace(transforms=[CustomCollect3D(keys=['img'])])
            base.modality = dict(use_camera=True)
            base.test_mode = True
            cams = {
                name: dict(data_path=f'/samples/{name}/frame.jpg',
                           sensor2lidar_rotation=np.eye(3),
                           sensor2lidar_translation=np.array([float(i), 0., 0.]),
                           cam_intrinsic=np.diag([float(i + 1), float(i + 1), 1.]))
                for i, name in reversed(list(enumerate(order)))
            }
            base.data_infos = [dict(
                token='sample', lidar_path='lidar.bin', sweeps=[],
                ego2global_translation=[0., 0., 0.], ego2global_rotation=[1., 0., 0., 0.],
                prev='', next='', scene_token='scene', can_bus=np.zeros(18),
                frame_idx=0, timestamp=0, cams=cams)]

        with patch.object(self.Base, '__init__', init):
            dataset = self.Dataset(pc_range=[-1, -1, -1, 1, 1, 1],
                                   use_semantics=False, use_depth=False, camera_types=order)
            original = self.Dataset(pc_range=[-1, -1, -1, 1, 1, 1],
                                    use_semantics=False, use_depth=False)
        self.assertEqual(list(original.data_infos[0]['cams']), order[::-1])
        info = dataset.get_data_info(0)
        self.assertEqual(info['img_filename'], [f'/samples/{name}/frame.jpg' for name in order])
        for i in range(6):
            self.assertEqual(info['cam_intrinsic'][i][0, 0], i + 1)
            self.assertEqual(info['lidar2cam'][i][0, 3], -i)
            np.testing.assert_allclose(info['lidar2img'][i],
                                       info['cam_intrinsic'][i] @ info['lidar2cam'][i])
            np.testing.assert_array_equal(info['metric3d_lidar2img_raw'][i], info['lidar2img'][i])
            self.assertIsNot(info['metric3d_lidar2img_raw'][i], info['lidar2img'][i])

    def test_live_metric3d_maps_without_legacy_dataset(self):
        from mmcv.parallel import DataContainer as DC
        from projects.bevdiffuser.data_utils import (
            CustomNuScenesDiffusionDataset_seg_depth as LegacyDataset,
            Occ3DDiffusionNuScenesDataset,
        )
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CustomCollect3D, PadMultiViewImage

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'samples' / 'CAM_FRONT'
            directory.mkdir(parents=True)
            labels = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int8)
            labels.tofile(directory / 'frame_mask.bin')
            raw_depth = np.array([[2, np.nan, 12], [4, 5, -1]], dtype=np.float32)
            np.save(directory / 'frame.npy', raw_depth)
            meta = metadata(2, 3)
            meta['filename'] = [str(directory / 'frame.jpg')]
            padded = PadMultiViewImage(size=(3, 5))(dict(img=[np.zeros((2, 3, 3), dtype=np.float32)]))
            meta.update({key: padded[key] for key in ('ori_shape', 'img_shape', 'pad_shape')})
            self.assertEqual(meta['ori_shape'], [(2, 3, 3)])
            self.assertEqual(meta['img_shape'], [(3, 5, 3)])
            meta['lidar2img'][0, 0, 2] = 1  # Translate pixels to the right by one.

            def init(base, **kwargs):
                base.pipeline = SimpleNamespace(transforms=[CustomCollect3D(keys=['img'])])

            with patch.object(self.Base, '__init__', init), \
                    patch.object(LegacyDataset, '__init__', side_effect=AssertionError('legacy init')), \
                    patch.object(LegacyDataset, '__getitem__', side_effect=AssertionError('legacy item')):
                dataset = self.Dataset(
                    pc_range=[-1, -1, -1, 1, 1, 1], semantic_path=tmp, depth_path=tmp,
                    use_semantics=True, use_depth=True, seg_id_remap={1: 2, 2: 3},
                    depth_raw_shape=(2, 3), depth_range=(.1, 10),
                )
                with patch.object(self.Base, '__getitem__', return_value={'img_metas': DC({0: meta})}):
                    item = dataset[0]
            expected_labels = torch.tensor([[[-1, 2, 3, -1, -1],
                                             [-1, 3, 3, -1, -1],
                                             [-1, -1, -1, -1, -1]]])
            expected_depth = torch.tensor([[[0., 2., 0., 0., 0.],
                                            [0., 4., 5., 0., 0.],
                                            [0., 0., 0., 0., 0.]]])
            torch.testing.assert_close(item['seg_maps'].data, expected_labels)
            torch.testing.assert_close(item['depth_maps'].data, expected_depth)
            self.assertNotIn('semantic_bev_probabilities', item)
            # Occ3D borrows these loader methods; preserve their standalone use.
            occ = object.__new__(Occ3DDiffusionNuScenesDataset)
            occ.__dict__.update(dataset.__dict__)
            torch.testing.assert_close(occ.load_depth_from_filenames(meta['filename'], meta), expected_depth)
            torch.testing.assert_close(occ.load_segmaps(meta['filename'], meta, tmp), expected_labels)
            # Keep checks that prevent geometrically incorrect training targets.
            invalid_meta = dict(meta, lidar2img=[np.diag([1., 1., 2., 1.])])
            with self.assertRaisesRegex(ValueError, '3D frame change'):
                dataset.load_depth_from_filenames(meta['filename'], invalid_meta)
            np.save(directory / 'frame.npy', raw_depth[:, :2])
            with self.assertRaisesRegex(ValueError, 'Expected floating'):
                dataset.load_depth_from_filenames(meta['filename'], meta)

    def test_prediction_targets_and_conditional_dropout(self):
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler(num_train_timesteps=20, clip_sample=False)
        target, noise = torch.randn(2, 384, 2, 2), torch.randn(2, 384, 2, 2)
        times = torch.tensor([3, 15])
        noisy = scheduler.add_noise(target, noise, times)
        for kind in ('sample', 'epsilon', 'v_prediction'):
            scheduler.register_to_config(prediction_type=kind)
            prediction = diffusion_target(scheduler, target, noise, times)
            torch.testing.assert_close(predict_clean(scheduler, prediction, noisy, times), target, atol=1e-5, rtol=1e-5)
        condition = dict(seg_cond=None, semantic_prob=torch.ones(2, 17, 50, 50))
        self.assertEqual(drop_condition(condition, 1)['semantic_prob'].sum(), 0)
        self.assertGreater(condition['semantic_prob'].sum(), 0)

    def test_training_checkpoint_resume_matches_uninterrupted_accumulation(self):
        self._check_training_resume(use_global_cls=False)

    def test_cls_training_validation_and_checkpoint_resume(self):
        self._check_training_resume(use_global_cls=True)

    def test_tensorboard_training_platform(self):
        self._check_training_resume(use_global_cls=True, report_to='tensorboard')

    def _check_training_resume(self, use_global_cls, report_to='none'):
        from accelerate import Accelerator
        from diffusers import DDPMScheduler
        from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
        from mmcv.parallel import DataContainer as DC
        from safetensors.torch import save_file, load_file
        import train_bev_diffuser_seg_dino as trainer

        class ToyDataset(torch.utils.data.Dataset):
            flag = np.zeros(3, dtype=np.uint8)
            def __init__(self, test_mode=False):
                self.test_mode = test_mode
            def __len__(self):
                return 3
            def __getitem__(self, index):
                meta = metadata(16, 16)
                pixels = torch.full((1, 3, 16, 16), float(100 + index))
                prob = torch.zeros(17, 50, 50)
                prob[2] = 1
                common = dict(depth_maps=DC(torch.full((1, 16, 16), 2.75)), semantic_bev_probabilities=DC(prob))
                if self.test_mode:
                    return dict(img=[DC(pixels, stack=True)], img_metas=[DC(meta, cpu_only=True)], **common)
                return dict(img=DC(pixels[None], stack=True), img_metas=DC({0: meta}, cpu_only=True), **common)

        cls_calls = []

        class ToyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(384, 384, 1)
                self.use_global_cls = use_global_cls
                if self.use_global_cls:
                    self.cls_proj = nn.Linear(384, 384)
            def forward(self, x, timesteps, img_metas, **condition):
                output = self.conv(x)
                if self.use_global_cls:
                    cls = condition['cls_tokens']
                    assert not cls.requires_grad
                    emb = self.cls_proj(cls.mean(dim=1))
                    cls_calls.append(self.training)
                    if self.training:
                        emb = emb * condition['cls_keep_mask'][:, None]
                    else:
                        assert 'cls_keep_mask' not in condition
                    output = output + emb[:, :, None, None]
                else:
                    assert 'cls_tokens' not in condition
                return output
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                save_file(self.state_dict(), str(Path(path) / SAFETENSORS_WEIGHTS_NAME))

        extractors = []
        def make_extractor(**kwargs):
            extractor = DINOv3BEVExtractor(encoder=FrozenDINOv3Encoder(model=FakeDINO()), projector=kwargs['projector'])
            extractors.append(extractor)
            return extractor

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(trainer, 'Accelerator', side_effect=lambda **kw: Accelerator(cpu=True, **kw)), \
                patch.object(trainer, 'build_dataset', side_effect=lambda opts, **kw: ToyDataset(opts['test_mode'])), \
                patch.object(trainer, 'build_unet_v2', side_effect=lambda cfg: ToyUNet()), \
                patch.object(trainer, 'DINOv3BEVExtractor', side_effect=make_extractor), \
                patch.object(DDPMScheduler, 'from_pretrained', side_effect=lambda *a, **kw: DDPMScheduler()):
            config = Path(tmp) / 'test_config.py'
            stage_one_config(use_global_cls).dump(str(config))
            argv = ['train', '--bev_config', str(config), '--output_dir', tmp, '--max_train_steps', '3',
                    '--gradient_accumulation_steps', '2', '--checkpointing_steps', '1',
                    '--dataloader_num_workers', '0', '--validation_batches', '1', '--report_to', report_to]
            with patch.object(sys, 'argv', argv):
                trainer.train()
            filename = Path(tmp) / 'checkpoint-3' / 'unet' / SAFETENSORS_WEIGHTS_NAME
            expected = load_file(str(filename))
            with patch.object(sys, 'argv', argv + ['--resume_from_checkpoint', str(Path(tmp) / 'checkpoint-2')]):
                trainer.train()
            actual = load_file(str(filename))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            import json
            metrics = json.loads((Path(tmp) / 'checkpoint-3' / 'validation.json').read_text())
            self.assertGreater(metrics['observed_fraction'], 0)
            self.assertTrue(all(p.grad is None for model in extractors for p in model.parameters()))
            if report_to == 'tensorboard':
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                events = EventAccumulator(str(Path(tmp) / 'tensorboard_logs')).Reload()
                tags = set(events.Tags()['scalars'])
                self.assertTrue({'step/step_cnt', 'step/epoch', 'lr/learning_rate',
                                 'train/denoise_loss', 'train/total_loss', 'val/mse'} <= tags)
                self.assertEqual(max(event.step for event in events.Scalars('step/step_cnt')), 6)
                self.assertEqual(json.loads((Path(tmp) / 'run_args.json').read_text())['num_train_epochs'], 2)
            if use_global_cls:
                self.assertIn(True, cls_calls)
                self.assertIn(False, cls_calls)

    def test_actual_unet_384_io_and_backward_with_reduced_internal_width(self):
        from mmcv import Config
        from model_utils import build_unet_v2
        cfg = stage_one_config()
        self.assertEqual(cfg.data.train.queue_length, 1)
        self.assertTrue(cfg.data.train.load_depth_in_cache)
        self.assertEqual(cfg.unet.parameters.model_channels, 256)
        # Bind the real full-size config before reducing test memory/compute.
        import inspect
        from layout_diffusion.seg_diffusion_unet_v5 import DiffusionUNetModel
        inspect.signature(DiffusionUNetModel).bind(**cfg.unet.parameters)
        cfg.unet.parameters.update(model_channels=32, seg_channels=[32, 32, 32],
                                   num_res_blocks=1, attention_ds=[4])
        cfg.unet.parameters.seg_bev_aligner.emb_channels = 32
        model = build_unet_v2(cfg.unet)
        latent = torch.randn(1, 384, 50, 50)
        probabilities = torch.zeros(1, 17, 50, 50)
        probabilities[:, 2] = 1
        pred = model(latent, torch.tensor([100]), [{}], semantic_prob=probabilities)
        self.assertEqual(pred.shape, latent.shape)
        (pred - latent).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            load_unet_strict(model, tmp)
            other = nn.Conv2d(256, 256, 1)
            with self.assertRaises(RuntimeError):
                load_unet_strict(other, tmp)


class CLSGlobalConditionTest(unittest.TestCase):
    def build_model(self):
        import inspect
        from mmcv import Config
        from model_utils import build_unet_v2
        from layout_diffusion.seg_diffusion_unet_v5_dino import DiffusionUNetModel
        cfg = Config.fromfile(str(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_dino.py'))
        self.assertNotIn('use_global_seg', cfg.unet.parameters)
        self.assertTrue(cfg.unet.parameters.use_global_cls)
        self.assertEqual(cfg.unet.parameters.cls_channels, 384)
        self.assertEqual(cfg.unet.parameters.model_channels * 4, 1024)
        self.assertTrue(cfg.data.train.load_depth_in_cache)
        for split in ('train', 'val', 'test'):
            self.assertEqual(cfg.data[split].camera_types, cfg.camera_types)
        self.assertEqual(cfg.unet.parameters.cls_camera_order, cfg.camera_types)
        inspect.signature(DiffusionUNetModel).bind(**cfg.unet.parameters)
        cfg.unet.parameters.update(model_channels=32, seg_channels=[32, 32, 32],
                                   num_res_blocks=1, attention_ds=[4])
        cfg.unet.parameters.seg_bev_aligner.emb_channels = 32
        return build_unet_v2(cfg.unet)

    def test_concat_camera_order(self):
        model = self.build_model()
        self.assertEqual(model.cls_global_embed[0].normalized_shape, (2304,))
        self.assertEqual(model.cls_global_embed[1].in_features, 2304)
        cls = torch.randn(2, 6, 384)
        seen = []
        hook = model.cls_global_embed.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            model.encode_global_cls(cls)
            model.encode_global_cls(cls.flip(1))
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], cls.flatten(1))
        torch.testing.assert_close(seen[1], cls.flip(1).flatten(1))
        self.assertFalse(torch.equal(seen[0], seen[1]))

    def test_shared_dropout_and_condition_attachment(self):
        cls = torch.randn(3, 6, 384)
        for key in ('semantic_prob', 'seg_cond'):
            condition = {key: torch.ones(3, 2, 4, 4)}
            self.assertNotIn('cls_tokens', attach_cls_condition(condition, {}, False))
            attached = attach_cls_condition(condition, {'cls_tokens': cls}, True)
            self.assertNotIn('cls_tokens', condition)
            with patch('train_bev_diffuser_seg_dino.torch.rand', return_value=torch.tensor([0.1, 0.9, 0.3])):
                dropped = drop_condition(attached, 0.2)
            torch.testing.assert_close(dropped['cls_keep_mask'], torch.tensor([False, True, True]))
            torch.testing.assert_close(dropped[key][:, 0, 0, 0], torch.tensor([0., 1., 1.]))
            self.assertIs(dropped['cls_tokens'], cls)
            torch.testing.assert_close(attached[key], torch.ones_like(attached[key]))
            self.assertNotIn('cls_keep_mask', attached)
            self.assertFalse(drop_condition(attached, 1)['cls_keep_mask'].any())
            self.assertTrue(drop_condition(attached, 0)['cls_keep_mask'].all())

    def test_cls_embedding_initialization_gradients_and_mask(self):
        model = self.build_model()
        cls = torch.randn(2, 6, 384)
        emb = model.encode_global_cls(cls)
        self.assertEqual(emb.shape, (2, 128))
        torch.testing.assert_close(emb, torch.zeros_like(emb))
        # First update reaches the zero-initialized last projection; the next
        # backward reaches the earlier MLP layers through that learned weight.
        optimizer = torch.optim.SGD(model.cls_global_embed.parameters(), lr=0.1)
        target = torch.randn_like(emb)
        (emb - target).square().mean().backward()
        self.assertGreater(model.cls_global_embed[-1].weight.grad.abs().sum(), 0)
        optimizer.step()
        optimizer.zero_grad()
        (model.encode_global_cls(cls) - target).square().mean().backward()
        self.assertGreater(model.cls_global_embed[1].weight.grad.abs().sum(), 0)
        with torch.no_grad():
            model.cls_global_embed[-1].bias.fill_(1)
            a = model.encode_global_cls(cls)
            b = model.encode_global_cls(cls.flip(1))
            self.assertFalse(torch.allclose(a, b))
            self.assertFalse(torch.allclose(a, model.encode_global_cls(torch.randn_like(cls))))
            masked = model.encode_global_cls(cls, torch.tensor([False, True]))
            torch.testing.assert_close(masked[0], torch.zeros_like(masked[0]))
            torch.testing.assert_close(masked[1], a[1])

    def test_cls_forward_spatial_independence_frozen_encoder_and_checkpoint(self):
        model = self.build_model()
        encoder = FrozenDINOv3Encoder(model=FakeDINO())
        metas = [metadata(16, 16, 6)]
        metas[0]['filename'] = [f'/samples/{camera}/frame.jpg' for camera in model.cls_camera_order]
        cls = encoder(torch.ones(1, 6, 3, 16, 16) * 127, metas)['cls_tokens']
        self.assertFalse(cls.requires_grad)
        # Give zero-initialized residual/output projections nonzero test weights
        # so the forward dependency is observable without a training warm-up.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2 and torch.count_nonzero(parameter) == 0:
                    nn.init.normal_(parameter, std=0.01)
        latent = torch.randn(1, 384, 50, 50)
        probabilities = torch.zeros(1, 17, 50, 50)
        probabilities[:, 2] = 1
        other_probabilities = torch.zeros_like(probabilities)
        other_probabilities[:, 3] = 1
        embeddings = []
        hook = model.input_blocks[0].register_forward_pre_hook(
            lambda module, args: embeddings.append(args[1].detach().clone()))
        with torch.no_grad():
            pred = model(latent, torch.tensor([100]), metas, semantic_prob=probabilities, cls_tokens=cls)
            other = model(latent, torch.tensor([100]), metas, semantic_prob=other_probabilities, cls_tokens=cls)
            changed_cls = model(latent, torch.tensor([100]), metas, semantic_prob=probabilities,
                                cls_tokens=torch.randn_like(cls))
        hook.remove()
        torch.testing.assert_close(embeddings[0], embeddings[1])
        self.assertFalse(torch.allclose(embeddings[0], embeddings[2]))
        self.assertFalse(torch.allclose(pred, other))
        self.assertFalse(torch.allclose(pred, changed_cls))
        self.assertEqual(pred.shape, latent.shape)
        pred = model(latent, torch.tensor([100]), metas, semantic_prob=probabilities, cls_tokens=cls)
        (pred - latent).square().mean().backward()
        self.assertGreater(model.cls_global_embed[-1].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            expected = model.encode_global_cls(cls).detach().clone()
            model.save_pretrained(tmp)
            with torch.no_grad():
                model.cls_global_embed[-1].bias.add_(1)
            load_unet_strict(model, tmp)
            torch.testing.assert_close(model.encode_global_cls(cls), expected)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
