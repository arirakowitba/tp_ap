"""Unit and multiprocess integration tests for atomic project coordination lock (CX-0008 / CX-0010)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.coordination_lock import ProjectLock, is_process_alive

REPO_ROOT = Path(__file__).resolve().parent.parent


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
        self.assertFalse(lock2.acquire())
        self.assertFalse(lock2.release())
        self.assertTrue(lock1.is_locked())

        self.assertTrue(lock1.release())

        self.assertTrue(lock2.acquire())
        info = lock2.read_lock_info()
        self.assertEqual(info.owner_id, "agent_2")
        self.assertTrue(lock2.release())

    def test_cx0010_deterministic_interleaving_race(self):
        """Regression test for CX-0010: contender B enters acquire() between A's mkdir and _write_lock_info."""
        a = ProjectLock(lock_dir=self.lock_dir, owner_id="a")
        b = ProjectLock(lock_dir=self.lock_dir, owner_id="b")
        orig = a._write_lock_info
        results = {}

        def delayed_write():
            results["b"] = b.acquire()
            orig()

        a._write_lock_info = delayed_write
        results["a"] = a.acquire()

        self.assertTrue(results["a"])
        self.assertFalse(results["b"])
        info = a.read_lock_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.owner_id, "a")
        # Contender B must NOT be able to release A's lock
        self.assertFalse(b.release())
        self.assertTrue(a.is_locked())
        self.assertTrue(a.release())

    def test_empty_initializing_lock_dir_not_deleted(self):
        """Never delete an empty lock directory that is within its initialization grace window."""
        os.mkdir(self.lock_dir)
        contender = ProjectLock(
            lock_dir=self.lock_dir, owner_id="contender", empty_dir_stale_seconds=10.0
        )
        self.assertFalse(contender.acquire())
        # Directory must still exist untouched
        self.assertTrue(self.lock_dir.exists())

    def test_same_owner_id_cannot_release_other_instance_lock(self):
        """Two instances with identical owner_id in the same PID cannot release each other."""
        a = ProjectLock(lock_dir=self.lock_dir, owner_id="shared_owner")
        b = ProjectLock(lock_dir=self.lock_dir, owner_id="shared_owner")
        self.assertTrue(a.acquire())
        self.assertFalse(b.acquire())
        self.assertFalse(b.release())
        self.assertTrue(a.is_locked())
        self.assertTrue(a.release())

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
            "description": "crashed",
        }
        (self.lock_dir / "lock_info.json").write_text(json.dumps(fake_info))

        self.assertFalse(is_process_alive(9999999))

        lock = ProjectLock(lock_dir=self.lock_dir, owner_id="recovery_agent")
        self.assertTrue(lock.acquire())
        info = lock.read_lock_info()
        self.assertEqual(info.owner_id, "recovery_agent")
        self.assertEqual(info.host_pid, os.getpid())
        self.assertTrue(lock.release())

    def test_multiprocess_contention_and_crash_recovery(self):
        """Spawns a child process holding the lock, verifies contention, kills child, verifies recovery."""
        ready_file = self.temp_dir / "child_ready.txt"
        child_code = (
            "import sys, time; "
            f"sys.path.insert(0, {repr(str(REPO_ROOT))}); "
            "from pathlib import Path; "
            "from src.coordination_lock import ProjectLock; "
            f"lock = ProjectLock(lock_dir=Path({repr(str(self.lock_dir))}), owner_id='child_proc'); "
            "assert lock.acquire(), 'child failed to acquire'; "
            f"Path({repr(str(ready_file))}).write_text('ready'); "
            "time.sleep(30)"
        )
        proc = subprocess.Popen([sys.executable, "-c", child_code])
        try:
            deadline = time.time() + 10.0
            while not ready_file.exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready_file.exists(), "Child process did not acquire lock in time")

            # While child is alive and holding OS lock, parent must NOT acquire
            parent_contender = ProjectLock(lock_dir=self.lock_dir, owner_id="parent_contender")
            self.assertFalse(parent_contender.acquire())

            # Simulate hard crash of child process
            proc.kill()
            proc.wait(timeout=5)

            # Wait briefly for OS process table cleanup
            deadline = time.time() + 5.0
            while is_process_alive(proc.pid) and time.time() < deadline:
                time.sleep(0.05)

            # Now parent recovery lock must cleanly recover and acquire
            recovery = ProjectLock(lock_dir=self.lock_dir, owner_id="parent_recovery")
            self.assertTrue(recovery.acquire())
            info = recovery.read_lock_info()
            self.assertEqual(info.owner_id, "parent_recovery")
            self.assertEqual(info.host_pid, os.getpid())
            self.assertTrue(recovery.release())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_two_concurrent_runners_and_duplicate_task_prevention(self):
        """Tests two concurrent run_coordination_check.py processes + duplicate task guard."""
        log_file = self.temp_dir / "runner_log.jsonl"
        marker_file = self.temp_dir / "task_started.txt"
        counter_file = self.temp_dir / "task_count.txt"
        state_file = self.temp_dir / "state.json"
        state_file.write_text(json.dumps({"last_processed_codex_message": "CX-0007"}))

        dummy_script = self.temp_dir / "dummy_task.py"
        dummy_script.write_text(
            "import time\n"
            "from pathlib import Path\n"
            f"m = Path({repr(str(marker_file))})\n"
            f"c = Path({repr(str(counter_file))})\n"
            "n = int(c.read_text()) if c.exists() else 0\n"
            "c.write_text(str(n + 1))\n"
            "m.write_text('started')\n"
            "time.sleep(1.2)\n",
            encoding="utf-8",
        )
        task_cmd = f"{sys.executable} {dummy_script}"

        runner_script = str(REPO_ROOT / "src" / "run_coordination_check.py")
        p1 = subprocess.Popen(
            [
                sys.executable,
                runner_script,
                "--lock-dir",
                str(self.lock_dir),
                "--owner",
                "runner_1",
                "--skip-fetch",
                "--state-file",
                str(state_file),
                "--log-file",
                str(log_file),
                "--task-id",
                "T_CONCURRENT_TEST",
                "--task-cmd",
                task_cmd,
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait until runner_1 is inside the task critical section
            deadline = time.time() + 10.0
            while not marker_file.exists() and p1.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if not marker_file.exists():
                out_err = p1.communicate(timeout=5)
                self.fail(f"Runner 1 did not start task in time. Output: {out_err}")

            # Launch runner_2 concurrently while runner_1 holds the lock
            p2 = subprocess.run(
                [
                    sys.executable,
                    runner_script,
                    "--lock-dir",
                    str(self.lock_dir),
                    "--owner",
                    "runner_2",
                    "--skip-fetch",
                    "--state-file",
                    str(state_file),
                    "--log-file",
                    str(log_file),
                    "--task-id",
                    "T_CONCURRENT_TEST",
                    "--task-cmd",
                    task_cmd,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(p2.returncode, 0)
            out2 = json.loads(p2.stdout)
            self.assertEqual(out2["status"], "SKIPPED_LOCK_HELD")
            self.assertEqual(out2["held_by"]["owner_id"], "runner_1")

            # Wait for runner_1 to finish
            stdout1, stderr1 = p1.communicate(timeout=15)
            self.assertEqual(p1.returncode, 0, msg=stderr1)
            out1 = json.loads(stdout1)
            self.assertEqual(out1["status"], "COMPLETED")
            self.assertEqual(out1["task_exit_code"], 0)
            self.assertIn("CX-0011", out1["pending_codex_messages"])

            # Task should have executed exactly once
            self.assertEqual(int(counter_file.read_text()), 1)

            # Launch runner_3 after runner_1 completed with the same task-id -> must skip duplicate
            p3 = subprocess.run(
                [
                    sys.executable,
                    runner_script,
                    "--lock-dir",
                    str(self.lock_dir),
                    "--owner",
                    "runner_3",
                    "--skip-fetch",
                    "--state-file",
                    str(state_file),
                    "--log-file",
                    str(log_file),
                    "--task-id",
                    "T_CONCURRENT_TEST",
                    "--task-cmd",
                    task_cmd,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(p3.returncode, 0)
            out3 = json.loads(p3.stdout)
            self.assertEqual(out3["status"], "SKIPPED_DUPLICATE_TASK")
            self.assertEqual(int(counter_file.read_text()), 1)
        finally:
            if p1.poll() is None:
                p1.kill()
                p1.wait(timeout=5)

    def test_heartbeat_update(self):
        lock = ProjectLock(lock_dir=self.lock_dir, owner_id="agent_hb")
        self.assertTrue(lock.acquire())
        info1 = lock.read_lock_info()
        time.sleep(0.05)
        self.assertTrue(lock.heartbeat())
        info2 = lock.read_lock_info()
        self.assertGreaterEqual(info2.heartbeat_utc, info1.heartbeat_utc)
        self.assertTrue(lock.release())

    def test_cx0014_large_stdout_stderr_no_pipe_deadlock(self):
        """CX-0014: Child writing > 512 KB to both stdout and stderr must never deadlock on OS pipe buffers."""
        log_file = Path(self.temp_dir) / "runner_large_io.jsonl"
        state_file = Path(self.temp_dir) / "state_large_io.json"
        state_file.write_text(json.dumps({"last_processed_codex_message": "CX-0014"}), encoding="utf-8")

        noisy_script = Path(self.temp_dir) / "noisy_task.py"
        noisy_script.write_text(
            "import sys\n"
            "chunk = 'A' * 8192 + '\\n'\n"
            "for _ in range(80):\n"
            "    sys.stdout.write(chunk)\n"
            "    sys.stderr.write(chunk)\n"
            "sys.stdout.flush()\n"
            "sys.stderr.flush()\n",
            encoding="utf-8",
        )
        runner_script = str(REPO_ROOT / "src" / "run_coordination_check.py")
        task_cmd = f'"{sys.executable}" "{noisy_script}"'

        res = subprocess.run(
            [
                sys.executable,
                runner_script,
                "--lock-dir",
                str(self.lock_dir),
                "--owner",
                "runner_noisy",
                "--skip-fetch",
                "--state-file",
                str(state_file),
                "--log-file",
                str(log_file),
                "--task-id",
                "T_LARGE_OUTPUT",
                "--task-cmd",
                task_cmd,
                "--task-timeout-seconds",
                "15",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        out = json.loads(res.stdout)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertEqual(out["task_exit_code"], 0)
        self.assertGreater(out["stdout_bytes"], 500_000)
        self.assertGreater(out["stderr_bytes"], 500_000)

    def test_cx0014_task_timeout_kills_process_tree_and_not_marked_completed(self):
        """CX-0014: Hanging task must be killed (with child tree) on timeout, logged as TIMED_OUT, and not marked COMPLETED."""
        log_file = Path(self.temp_dir) / "runner_timeout.jsonl"
        state_file = Path(self.temp_dir) / "state_timeout.json"
        state_file.write_text(json.dumps({"last_processed_codex_message": "CX-0014"}), encoding="utf-8")

        hang_script = Path(self.temp_dir) / "hang_task.py"
        hang_script.write_text(
            "import time, sys\n"
            "print('starting hang task', flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        runner_script = str(REPO_ROOT / "src" / "run_coordination_check.py")
        task_cmd = f'"{sys.executable}" "{hang_script}"'

        t0 = time.monotonic()
        res = subprocess.run(
            [
                sys.executable,
                runner_script,
                "--lock-dir",
                str(self.lock_dir),
                "--owner",
                "runner_timeout",
                "--skip-fetch",
                "--state-file",
                str(state_file),
                "--log-file",
                str(log_file),
                "--task-id",
                "T_TIMEOUT_TEST",
                "--task-cmd",
                task_cmd,
                "--task-timeout-seconds",
                "0.8",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 10.0)
        self.assertEqual(res.returncode, 124)
        out = json.loads(res.stdout)
        self.assertEqual(out["status"], "TIMED_OUT")
        self.assertEqual(out["task_exit_code"], 124)

        # Lock must be released after timeout
        check_lock = ProjectLock(lock_dir=self.lock_dir, owner_id="post_timeout_verifier")
        self.assertTrue(check_lock.acquire())
        self.assertTrue(check_lock.release())

        # Timed-out task must NOT be treated as completed (a subsequent fast run with same task_id must execute)
        fast_script = Path(self.temp_dir) / "fast_task.py"
        fast_script.write_text("print('recovered ok')\n", encoding="utf-8")
        res2 = subprocess.run(
            [
                sys.executable,
                runner_script,
                "--lock-dir",
                str(self.lock_dir),
                "--owner",
                "runner_retry",
                "--skip-fetch",
                "--state-file",
                str(state_file),
                "--log-file",
                str(log_file),
                "--task-id",
                "T_TIMEOUT_TEST",
                "--task-cmd",
                f'"{sys.executable}" "{fast_script}"',
                "--task-timeout-seconds",
                "10",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(res2.returncode, 0)
        out2 = json.loads(res2.stdout)
        self.assertEqual(out2["status"], "COMPLETED")

    def test_cx0014_cli_run_holds_and_releases_lock(self):
        """CX-0014: CLI 'run --cmd ...' holds lock during command and releases on exit."""
        lock_script = str(REPO_ROOT / "src" / "coordination_lock.py")
        marker = Path(self.temp_dir) / "cli_run_ok.txt"
        cmd = f'"{sys.executable}" -c "from pathlib import Path; Path(r\'{marker}\').write_text(\'ok\')"'
        res = subprocess.run(
            [
                sys.executable,
                lock_script,
                "run",
                "--lock-dir",
                str(self.lock_dir),
                "--owner",
                "cli_runner",
                "--cmd",
                cmd,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertTrue(marker.exists())
        self.assertFalse(self.lock_dir.exists())


if __name__ == "__main__":
    unittest.main()
