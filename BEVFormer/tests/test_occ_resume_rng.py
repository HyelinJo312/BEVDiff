"""Focused CPU checks for occupancy Stage-1 checkpoint RNG restoration."""

import ast
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


TRAINER_PATH = (Path(__file__).resolve().parents[1] /
                "projects/bevdiffuser/train_bev_diffuser_occ_only_seg.py")
TRAINER_TREE = ast.parse(TRAINER_PATH.read_text())


def _load_restore_helper():
    """Load only the helper so tests do not import the GPU training plugin."""
    helper = next(node for node in TRAINER_TREE.body
                  if isinstance(node, ast.FunctionDef) and
                  node.name == "_restore_resume_random_state")
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace = {"os": os, "random": random, "np": np, "torch": torch}
    exec(compile(module, str(TRAINER_PATH), "exec"), namespace)
    return namespace[helper.name]


restore_rng = _load_restore_helper()


class OccResumeRandomStateTests(unittest.TestCase):
    def setUp(self):
        self.original_states = (random.getstate(), np.random.get_state(),
                                torch.get_rng_state())
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self._restore_original_states)

    def _restore_original_states(self):
        python_state, numpy_state, torch_state = self.original_states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

    def _write_rank_state(self, rank, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        states = {
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
            "torch_cuda_manual_seed": [torch.arange(4, dtype=torch.uint8)],
        }
        path = os.path.join(self.temp_dir.name, f"random_states_{rank}.pkl")
        torch.save(states, path)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        return path, expected, states

    def test_restores_exact_python_numpy_and_torch_streams_from_selected_rank(self):
        self._write_rank_state(0, 123)
        rank_path, expected, _ = self._write_rank_state(1, 456)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        with patch.object(torch.cuda, "is_available", return_value=False):
            restored_path = restore_rng(self.temp_dir.name, 1)

        self.assertEqual(restored_path, rank_path)
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected)

    def test_missing_and_corrupt_rank_state_fail_with_rank_and_path(self):
        self._write_rank_state(0, 123)
        with self.assertRaisesRegex(FileNotFoundError, "rank 1.*random_states_1.pkl"):
            restore_rng(self.temp_dir.name, 1)

        corrupt_path = os.path.join(self.temp_dir.name, "random_states_1.pkl")
        Path(corrupt_path).write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(RuntimeError, "rank 1.*random_states_1.pkl"):
            restore_rng(self.temp_dir.name, 1)

    def test_incomplete_rank_state_fails(self):
        path, _, states = self._write_rank_state(3, 234)
        del states["torch_cuda_manual_seed"]
        torch.save(states, path)
        with self.assertRaisesRegex(RuntimeError, "rank 3.*random_states_3.pkl"):
            restore_rng(self.temp_dir.name, 3)

    def test_available_cuda_restores_saved_states_without_gpu(self):
        _, _, saved = self._write_rank_state(2, 789)
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "set_rng_state_all") as set_cuda_rng:
            restore_rng(self.temp_dir.name, 2)
        set_cuda_rng.assert_called_once()
        self.assertTrue(torch.equal(set_cuda_rng.call_args.args[0][0],
                                    saved["torch_cuda_manual_seed"][0]))

    def test_resume_calls_helper_immediately_after_accelerate_load(self):
        train = next(node for node in TRAINER_TREE.body
                     if isinstance(node, ast.FunctionDef) and node.name == "train")
        resume_blocks = [node for node in ast.walk(train)
                         if isinstance(node, ast.If) and
                         isinstance(node.test, ast.Attribute) and
                         node.test.attr == "resume_from_checkpoint"]
        calls = [node.body for node in resume_blocks
                 if any(isinstance(stmt, ast.Expr) and
                        isinstance(stmt.value, ast.Call) and
                        isinstance(stmt.value.func, ast.Attribute) and
                        stmt.value.func.attr == "load_state" for stmt in node.body)]
        self.assertEqual(len(calls), 1)
        statements = calls[0]
        load_index = next(i for i, stmt in enumerate(statements)
                          if isinstance(stmt, ast.Expr) and
                          isinstance(stmt.value, ast.Call) and
                          isinstance(stmt.value.func, ast.Attribute) and
                          stmt.value.func.attr == "load_state")
        restore = statements[load_index + 1]
        self.assertIsInstance(restore, ast.Assign)
        self.assertEqual(restore.value.func.id, "_restore_resume_random_state")
        self.assertEqual([ast.unparse(arg) for arg in restore.value.args],
                         ["resume_path", "accelerator.process_index"])


if __name__ == "__main__":
    unittest.main()
