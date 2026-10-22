"""Check evaluation setup without loading nuScenes data or CUDA plugins."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mmcv import Config
from mmcv.utils import Registry, build_from_cfg


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/test.py'


def load_build_dataset(builder):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == '_build_dataset')
    scope = {'build_dataset': builder}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['_build_dataset']


class DatasetSetupTests(unittest.TestCase):
    def test_actual_segmentation_config_has_no_diffusion_defaults(self):
        cfg = Config.fromfile(str(ROOT / 'projects/configs/bevformer/bevformer_tiny_seg_v3.py'))
        cfg.data.test.test_mode = True
        builder = Mock()
        load_build_dataset(builder)(cfg)
        builder.assert_called_once_with(cfg.data.test, default_args=None)

    def test_strict_segmentation_constructor_accepts_setup(self):
        registry = Registry('test_datasets')

        @registry.register_module(name='CustomNuScenesDataset')
        class SegDataset:
            def __init__(self, test_mode):
                self.test_mode = test_mode

        builder = lambda cfg, default_args: build_from_cfg(cfg, registry, default_args)
        cfg = Config(dict(point_cloud_range=[-10, -10, -2, 10, 10, 2],
                          num_classes=7, data=dict(test=dict(
                              type='CustomNuScenesDataset', test_mode=True))))
        self.assertTrue(load_build_dataset(builder)(cfg).test_mode)

    def test_diffusion_defaults_and_explicit_dataset_override(self):
        registry = Registry('test_datasets')

        @registry.register_module(name='CustomNuScenesDiffusionDataset_layout_seg')
        class DiffusionDataset:
            def __init__(self, pc_range, num_classes, use_3d_bbox, num_bboxes):
                self.values = (pc_range, num_classes, use_3d_bbox, num_bboxes)

        builder = lambda cfg, default_args: build_from_cfg(cfg, registry, default_args)
        cfg = Config(dict(point_cloud_range=[-1, -1, -1, 1, 1, 1],
                          num_classes=7, use_3d_bbox=False, num_bboxes=10,
                          data=dict(test=dict(
                              type='CustomNuScenesDiffusionDataset_layout_seg',
                              num_classes=9))))
        result = load_build_dataset(builder)(cfg)
        self.assertEqual(result.values, (cfg.point_cloud_range, 9, False, 10))

    def test_list_of_segmentation_datasets_gets_no_diffusion_defaults(self):
        cfg = Config(dict(point_cloud_range=[0] * 6, data=dict(test=[
            dict(type='CustomNuScenesDataset'), dict(type='CustomNuScenesDataset')])))
        builder = Mock()
        load_build_dataset(builder)(cfg)
        builder.assert_called_once_with(cfg.data.test, default_args=None)


class SingleGPUEvaluationSetupTests(unittest.TestCase):
    def test_model_moves_to_gpu_before_wrapping(self):
        tree = ast.parse(SOURCE.read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        branch = next(node for node in main.body
                      if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                      and isinstance(node.test.operand, ast.Name)
                      and node.test.operand.id == 'distributed')
        setup = compile(ast.Module(body=branch.body, type_ignores=[]), str(SOURCE), 'exec')
        model, factory, inference = Mock(), Mock(), Mock()
        scope = dict(model=model, MMDataParallel=factory, single_gpu_test=inference,
                     data_loader=Mock(), args=SimpleNamespace(show=False, show_dir=None))
        exec(setup, scope)
        model.cuda.assert_called_once_with(0)
        factory.assert_called_once_with(model.cuda.return_value, device_ids=[0])
        self.assertIs(inference.call_args.args[0], factory.return_value)


class DistributedEvaluationSetupTests(unittest.TestCase):
    def test_compatibility_flag_is_set_before_inference_and_preserved(self):
        tree = ast.parse(SOURCE.read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        branch = next(node for node in main.body
                      if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                      and isinstance(node.test.operand, ast.Name)
                      and node.test.operand.id == 'distributed')
        setup = compile(ast.Module(body=branch.orelse, type_ignores=[]), str(SOURCE), 'exec')
        for wrapper in (SimpleNamespace(),
                        SimpleNamespace(_use_replicated_tensor_module=True)):
            expected = getattr(wrapper, '_use_replicated_tensor_module', False)

            def check_flag(model, *args):
                self.assertIs(model._use_replicated_tensor_module, expected)

            scope = dict(model=Mock(), MMDistributedDataParallel=Mock(return_value=wrapper),
                         torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
                         data_loader=Mock(), args=SimpleNamespace(tmpdir=None, gpu_collect=False),
                         multi_gpu_test=check_flag)
            exec(setup, scope)


if __name__ == '__main__':
    unittest.main()
