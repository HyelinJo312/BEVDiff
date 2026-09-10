"""Writer exclusion and CLI diagnostics without CUDA or the training stack."""

import errno
import fcntl
import importlib.util
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


BUILDER = Path(__file__).resolve().parents[1] / 'projects/bevdiffuser/build_semantic_bev_cache.py'
spec = importlib.util.spec_from_file_location('cache_builder', BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

# Run the real CLI/locking path, substituting only the expensive build body.
CHILD = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('cache_builder', sys.argv.pop(1))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
def build_cache(args, parser):
    assert 'torch' not in sys.modules
    print('TEST_BUILD_STARTED', flush=True)
builder.build_cache = build_cache
builder.main()
"""


class SemanticBEVCacheLockTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock_path = self.root / '.builder.lock'

    def start_waiter(self):
        child = subprocess.Popen(
            [sys.executable, '-u', '-c', CHILD, str(BUILDER),
             '--cache-root', str(self.root), '--wait-for-lock'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)

        self.addCleanup(cleanup)
        self.assertTrue(select.select([child.stdout], [], [], 10)[0])
        self.assertIn('Waiting for the lock; no GPU has been initialized', child.stdout.readline())
        self.assertIsNone(child.poll())
        return child

    def test_existing_file_is_reusable_and_never_replaced(self):
        self.lock_path.write_text('existing lock file')
        inode = self.lock_path.stat().st_ino
        for _ in range(2):
            with builder.cache_builder_lock(self.root):
                self.assertEqual(self.lock_path.stat().st_ino, inode)
        self.assertEqual(self.lock_path.read_text(), 'existing lock file')

    def test_exception_releases_lock_without_unlink(self):
        with self.assertRaisesRegex(RuntimeError, 'build failed'):
            with builder.cache_builder_lock(self.root):
                raise RuntimeError('build failed')
        inode = self.lock_path.stat().st_ino
        with builder.cache_builder_lock(self.root):
            self.assertEqual(self.lock_path.stat().st_ino, inode)

    def test_legacy_empty_lock_blocks_new_writer(self):
        with self.lock_path.open('a') as legacy:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(builder.CacheBuildLockedError, '--wait-for-lock'):
                with builder.cache_builder_lock(self.root):
                    self.fail('A second writer entered the critical section')
            self.assertEqual(self.lock_path.read_bytes(), b'')

    def test_real_cli_conflict_is_friendly_and_does_not_modify_cache(self):
        manifest = self.root / 'manifest.json'
        manifest.write_text('{"entries": {}}')
        with builder.cache_builder_lock(self.root):
            result = subprocess.run(
                [sys.executable, str(BUILDER), '--cache-root', str(self.root),
                 '--split', 'all', '--device', 'cuda'],
                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Another cache builder holds', result.stderr)
        self.assertIn('even on different GPUs', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertEqual(manifest.read_text(), '{"entries": {}}')
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {'manifest.json', '.builder.lock'})

    def test_waiter_builds_only_after_legacy_writer_releases(self):
        with self.lock_path.open('a') as legacy:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inode = self.lock_path.stat().st_ino
            child = self.start_waiter()
        stdout, stderr = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertIn('Cache lock acquired', stdout)
        self.assertIn('TEST_BUILD_STARTED', stdout)
        self.assertEqual(self.lock_path.stat().st_ino, inode)

    def test_cancel_waiter_preserves_current_owner(self):
        with builder.cache_builder_lock(self.root):
            child = self.start_waiter()
            child.send_signal(signal.SIGINT)
            stdout, stderr = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 130, stderr)
            self.assertNotIn('TEST_BUILD_STARTED', stdout)
            self.assertIn('interrupted', stderr)
            self.assertNotIn('Traceback', stderr)
            with self.assertRaises(builder.CacheBuildLockedError):
                with builder.cache_builder_lock(self.root):
                    self.fail('Cancelling the waiter released the original writer lock')

    def test_other_lock_errors_are_not_reported_as_contention(self):
        with patch.object(builder.fcntl, 'flock', side_effect=OSError(errno.EIO, 'I/O error')):
            with self.assertRaises(OSError) as caught:
                with builder.cache_builder_lock(self.root):
                    self.fail('Acquired a lock after an I/O error')
        self.assertEqual(caught.exception.errno, errno.EIO)

    def test_help_does_not_require_training_dependencies(self):
        result = subprocess.run([sys.executable, '-S', str(BUILDER), '--help'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--wait-for-lock', result.stdout)


if __name__ == '__main__':
    unittest.main()
