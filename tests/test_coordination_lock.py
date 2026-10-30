"""Unit tests for atomic project coordination lock."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from src.coordination_lock import ProjectLock, is_process_alive


class TestProjectLock(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.lock_dir = self.temp_dir / ".test_lock"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_acquire_and_release(self):
        lock = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_1")
        self.assertTrue(lock.acquire())
        self.assertTrue(self.lock_dir.exists())
        self.assertTrue((self.lock_dir / "lock_info.json").exists())

        info = lock.read_lock_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.owner_id, "agent_1")
        self.assertEqual(info.host_pid, os.getpid())

        self.assertTrue(lock.release())
        self.assertFalse(self.lock_dir.exists())

    def test_mutual_exclusion(self):
        lock1 = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_1")
        lock2 = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_2")

        self.assertTrue(lock1.acquire())
        # Second acquire must fail because lock1 is held by active process (current process)
        self.assertFalse(lock2.acquire())

        # Release lock1
        self.assertTrue(lock1.release())

        # Now lock2 should succeed
        self.assertTrue(lock2.acquire())
        info = lock2.read_lock_info()
        self.assertEqual(info.owner_id, "agent_2")
        self.assertTrue(lock2.release())

    def test_context_manager(self):
        with ProjectLock(lock_dir=self.lock_dir, owner_id="agent_cm") as lock:
            self.assertTrue(self.lock_dir.exists())
            self.assertTrue(lock.is_locked())

        self.assertFalse(self.lock_dir.exists())

    def test_context_manager_blocks_concurrent(self):
        with ProjectLock(lock_dir=self.lock_dir, owner_id="agent_cm1"):
            lock2 = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_cm2")
            with self.assertRaises(BlockingIOError):
                with lock2:
                    pass

    def test_stale_lock_recovery(self):
        # Create a fake lock with a non-existent PID (e.g. 9999999)
        os.mkdir(self.lock_dir)
        fake_info = {
            "owner_id": "crashed_agent",
            "host_pid": 9999999,
            "host_create_time": 0.0,
            "started_at_utc": "2026-10-30T00:00:00Z",
            "started_at_local": "2026-10-30T00:00:00-03:00",
            "heartbeat_utc": "2026-10-30T00:00:00Z",
            "heartbeat_local": "2026-10-30T00:00:00-03:00",
            "branch": "test",
            "description": "crashed"
        }
        (self.lock_dir / "lock_info.json").write_text(json.dumps(fake_info))

        # Ensure PID 9999999 is dead
        self.assertFalse(is_process_alive(9999999))

        # A new agent should detect stale lock, clean it up, and acquire
        lock = ProjectLock(lock_dir=self.lock_dir, owner_id="recovery_agent")
        self.assertTrue(lock.acquire())
        info = lock.read_lock_info()
        self.assertEqual(info.owner_id, "recovery_agent")
        self.assertEqual(info.host_pid, os.getpid())
        self.assertTrue(lock.release())

    def test_heartbeat_update(self):
        lock = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_hb")
        self.assertTrue(lock.acquire())
        info1 = lock.read_lock_info()
        time.sleep(0.05)
        self.assertTrue(lock.heartbeat())
        info2 = lock.read_lock_info()
        self.assertGreaterEqual(info2.heartbeat_utc, info1.heartbeat_utc)
        self.assertTrue(lock.release())


if __name__ == "__main__":
    unittest.main()
