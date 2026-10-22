"""Synthetic coordinator coverage without CUDA or training dependencies."""

from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BUILDER = Path(__file__).resolve().parents[1] / 'projects/bevdiffuser/build_semantic_bev_cache.py'
spec = importlib.util.spec_from_file_location('cache_builder_parallel', BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fake_worker(worker_id, mode, split, source_version, device, tasks, results):
    contract = {'source': source_version}
    if mode == 'contract':
        contract = {'source': 'different'}
    results.put(('ready', worker_id, contract, device))
    if mode == 'exit0':
        return
    if mode == 'exit0_one' and worker_id == 0:
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        index, token, cached = task
        if mode == 'die':
            os._exit(7)
        if mode == 'error':
            results.put(('error', worker_id, 'synthetic projection failure'))
            return
        value = None if cached else array('f', [float(index), float(worker_id)])
        results.put(('result', worker_id, index, {'sample_idx': token, 'geometry': index}, value))


class FakeDataset:
    def __init__(self, tokens):
        self.data_infos = [{'token': token} for token in tokens]

    def __len__(self):
        return len(self.data_infos)


class FakeCache:
    def __init__(self, root):
        self.root = Path(root)
        self.contract = {'source': 'synthetic'}
        self.output_frame = 'ego'
        self.manifest = {'entries': {}}
        self.saved_by = []

    def _path(self, token):
        return self.root / f'{token}.npy'

    def save(self, token, meta, value):
        if not isinstance(value, array) or value.typecode != 'f' or len(value) != 2:
            raise ValueError('Expected two float32 CPU values')
        self._path(token).write_bytes(value.tobytes())
        self.manifest['entries'][token] = {
            'geometry': meta['geometry'],
            'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
        }
        self.saved_by.append(os.getpid())

    def load(self, token, meta):
        entry = self.manifest['entries'][token]
        if entry['geometry'] != meta['geometry']:
            raise ValueError('geometry mismatch')
        data = self._path(token).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Corrupt semantic BEV cache')

    def flush(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))


class SemanticBEVCacheParallelTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = FakeCache(self.root)

    def run_jobs(self, tokens, devices=None, mode='normal'):
        jobs = builder.prepare_jobs(FakeDataset(tokens), self.cache, range(len(tokens)))
        with builder.cache_builder_lock(self.root):
            try:
                builder.run_workers(mode, 'train', 'synthetic', devices or ['cuda:0', 'cuda:1'],
                                    jobs, self.cache, 2, worker_target=fake_worker)
            finally:
                self.cache.flush()

    def test_shards_exact_once_and_only_coordinator_writes(self):
        self.run_jobs([f'token-{index}' for index in range(6)])
        self.assertEqual(len(self.cache.manifest['entries']), 6)
        self.assertEqual(self.cache.saved_by, [os.getpid()] * 6)
        self.assertEqual(len(list(self.root.glob('token-*.npy'))), 6)

    def test_resume_verifies_without_rewriting(self):
        self.run_jobs(['token-0', 'token-1'])
        before = [self.cache._path(token).stat().st_mtime_ns for token in ('token-0', 'token-1')]
        self.run_jobs(['token-0', 'token-1', 'token-2'])
        self.assertEqual(len(self.cache.saved_by), 3)
        self.assertEqual(before, [self.cache._path(token).stat().st_mtime_ns
                                  for token in ('token-0', 'token-1')])

    def test_corrupt_resume_rejects(self):
        self.run_jobs(['token-0'])
        self.cache._path('token-0').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            self.run_jobs(['token-0'])

    def test_orphan_and_duplicate_tokens_reject_before_workers(self):
        self.cache._path('orphan').write_bytes(b'existing')
        with self.assertRaisesRegex(FileExistsError, 'Orphan cache array'):
            builder.prepare_jobs(FakeDataset(['orphan']), self.cache, [0])
        with self.assertRaisesRegex(ValueError, 'Duplicate sample token'):
            builder.prepare_jobs(FakeDataset(['same', 'same']), self.cache, [0, 1])

    def test_worker_error_death_and_clean_exit_release_lock(self):
        for mode, message in [('error', 'synthetic projection failure'),
                              ('die', 'exited'),
                              ('exit0', 'exited')]:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_jobs(['token-0'], mode=mode)
                with builder.cache_builder_lock(self.root):
                    pass

    def test_contract_mismatch_rejects_and_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            self.run_jobs(['token-0'], mode='contract')
        with builder.cache_builder_lock(self.root):
            pass

    def test_one_worker_clean_exit_fails_while_other_results_arrive(self):
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            self.run_jobs([f'token-{index}' for index in range(30)], mode='exit0_one')
        with builder.cache_builder_lock(self.root):
            pass

    def test_launcher_rejects_duplicate_physical_gpu(self):
        launcher = BUILDER.with_name('build_semantic_bev_cache_occ.sh')
        result = subprocess.run(['bash', str(launcher), '--help'], capture_output=True,
                                text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,0'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('Duplicate physical GPU', result.stderr)

    def test_cli_device_and_index_validation(self):
        args, _ = builder.parse_args(['--cache-root', str(self.root), '--device', 'cuda'])
        self.assertEqual(args.selected_devices, ['cuda:0'])
        args, _ = builder.parse_args(['--cache-root', str(self.root), '--devices', 'cuda:3', 'cuda:01'])
        self.assertEqual(args.selected_devices, ['cuda:3', 'cuda:1'])
        for extra in (['--devices', 'cuda:0', 'cuda:00'], ['--devices', 'cuda:0,1'],
                      ['--device', 'cpu'], ['--split', 'train', '--indices', '0', '0'],
                      ['--split', 'train', '--indices', '-1']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                builder.parse_args(['--cache-root', str(self.root), *extra])


if __name__ == '__main__':
    unittest.main()
