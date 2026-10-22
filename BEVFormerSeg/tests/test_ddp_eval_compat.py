"""Exercise DDP setup without importing the dataset/CUDA plugin stack."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel


SOURCE = (Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/'
          'bevformer/apis/mmdet_train.py')


def wrap_models(model, eval_model=None, factory=None):
    # Execute the actual distributed setup branch, isolating it from training.
    tree = ast.parse(SOURCE.read_text())
    train = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == 'custom_train_detector')
    branch = next(node for node in train.body
                  if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name)
                  and node.test.id == 'distributed')
    setup = ast.Module(body=branch.body, type_ignores=[])
    scope = dict(model=model, eval_model=eval_model, cfg={},
                 torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
                 MMDistributedDataParallel=factory)
    exec(compile(setup, str(SOURCE), 'exec'), scope)
    return scope['model'], scope['eval_model']


class DDPEvaluationCompatibilityTests(unittest.TestCase):
    def test_missing_flag_on_both_wrappers(self):
        wrappers = [SimpleNamespace(), SimpleNamespace()]
        factory = Mock(side_effect=wrappers)
        model, eval_model = wrap_models(Mock(), Mock(), factory)
        self.assertIs(model._use_replicated_tensor_module, False)
        self.assertIs(eval_model._use_replicated_tensor_module, False)

    def test_existing_flags_are_preserved(self):
        wrappers = [SimpleNamespace(_use_replicated_tensor_module=True),
                    SimpleNamespace(_use_replicated_tensor_module=False)]
        model, eval_model = wrap_models(Mock(), Mock(), Mock(side_effect=wrappers))
        self.assertIs(model._use_replicated_tensor_module, True)
        self.assertIs(eval_model._use_replicated_tensor_module, False)

    def test_real_mmcv_evaluation_forward_on_cpu(self):
        # One process, CPU only: no training data, checkpoint, or GPU required.
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo', init_method=f'file://{tmp}/rendezvous',
                                    rank=0, world_size=1)
            try:
                module = torch.nn.Linear(2, 1)
                original = MMDistributedDataParallel(module, device_ids=[])
                original.eval()
                sample = torch.ones(1, 2)
                # Reproduce the reported failure when the installed PyTorch
                # no longer supplies MMCV's legacy flag.
                if not hasattr(original, '_use_replicated_tensor_module'):
                    with torch.no_grad(), self.assertRaisesRegex(
                            AttributeError, '_use_replicated_tensor_module'):
                        original(sample)
                wrapped, eval_model = wrap_models(
                    Mock(), factory=Mock(return_value=original))
                self.assertIsNone(eval_model)
                with torch.no_grad():
                    torch.testing.assert_close(wrapped(sample), module(sample))
            finally:
                dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
