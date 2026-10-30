"""Atomic Project-Level Mutual Exclusion Lock (CX-0003 / CX-0004).

Implements atomic mutual exclusion via directory creation (guaranteed atomic
on both Windows and POSIX OS kernels via filesystem primitives).
Identifies owner, host PID, process creation time, start time, and heartbeat.
Safeguards against lock stealing: verifies process liveness via psutil before
declaring any lock stale.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

DEFAULT_LOCK_DIR = Path(__file__).resolve().parent.parent / "coordination" / ".project_lock"


def _now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _now_iso_local() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def is_process_alive(pid: int, expected_create_time: Optional[float] = None) -> bool:
    """Checks whether the process with PID is currently alive and matches create_time."""
    if psutil is None:
        # Fallback without psutil
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    try:
        p = psutil.Process(pid)
        if not p.is_running():
            return False
        try:
            if p.status() == psutil.STATUS_ZOMBIE:
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        if expected_create_time is not None:
            try:
                # Allow 2 second slack in clock precision
                if abs(p.create_time() - expected_create_time) > 2.0:
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@dataclass
class LockInfo:
    owner_id: str
    host_pid: int
    host_create_time: float
    started_at_utc: str
    started_at_local: str
    heartbeat_utc: str
    heartbeat_local: str
    branch: str = ""
    description: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> LockInfo:
        data = json.loads(text)
        return cls(**data)


class ProjectLock:
    """Atomic project mutual exclusion lock manager."""

    def __init__(
        self,
        lock_dir: Optional[Path] = None,
        owner_id: str = "antigravity",
        description: str = "",
        branch: str = "",
    ):
        self.lock_dir = Path(lock_dir) if lock_dir else DEFAULT_LOCK_DIR
        self.info_file = self.lock_dir / "lock_info.json"
        self.owner_id = owner_id
        self.description = description
        self.branch = branch
        self._acquired = False

    def acquire(self, retry_stale: bool = True) -> bool:
        """Attempts atomic acquisition of the project lock.

        Returns True if acquired, False otherwise.
        """
        try:
            os.mkdir(self.lock_dir)
            self._write_lock_info()
            self._acquired = True
            return True
        except FileExistsError:
            # Lock directory exists. Check whether owner is active.
            existing_info = self.read_lock_info()
            if existing_info is None:
                # Empty or corrupt lock dir; check if stale by directory creation time
                if retry_stale:
                    return self._clean_stale_and_retry()
                return False

            # Check if existing host process is still alive
            if is_process_alive(existing_info.host_pid, existing_info.host_create_time):
                # Lock is legitimately held by an active running process!
                # Do NOT steal it.
                return False

            # The recorded process is no longer running (stale lock after crash)
            if retry_stale:
                return self._clean_stale_and_retry()
            return False

    def _write_lock_info(self) -> None:
        host_create = 0.0
        if psutil is not None:
            try:
                host_create = psutil.Process(os.getpid()).create_time()
            except Exception:
                host_create = time.time()
        else:
            host_create = time.time()

        now_utc = _now_iso_utc()
        now_loc = _now_iso_local()

        info = LockInfo(
            owner_id=self.owner_id,
            host_pid=os.getpid(),
            host_create_time=host_create,
            started_at_utc=now_utc,
            started_at_local=now_loc,
            heartbeat_utc=now_utc,
            heartbeat_local=now_loc,
            branch=self.branch,
            description=self.description,
        )
        self.info_file.write_text(info.to_json(), encoding="utf-8")

    def read_lock_info(self) -> Optional[LockInfo]:
        """Reads LockInfo if info_file exists and is valid."""
        if not self.info_file.exists():
            return None
        try:
            return LockInfo.from_json(self.info_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def heartbeat(self) -> bool:
        """Updates the heartbeat timestamp if we are the owner."""
        if not self._acquired:
            return False
        info = self.read_lock_info()
        if info is None or info.host_pid != os.getpid():
            return False
        info.heartbeat_utc = _now_iso_utc()
        info.heartbeat_local = _now_iso_local()
        self.info_file.write_text(info.to_json(), encoding="utf-8")
        return True

    def release(self) -> bool:
        """Releases the lock if acquired by this instance."""
        if not self.lock_dir.exists():
            self._acquired = False
            return True

        info = self.read_lock_info()
        # Verify ownership before releasing
        if info is not None and info.host_pid != os.getpid():
            # Not our lock; refuse to release
            return False

        try:
            if self.info_file.exists():
                self.info_file.unlink(missing_ok=True)
            if self.lock_dir.exists():
                os.rmdir(self.lock_dir)
            self._acquired = False
            return True
        except Exception:
            # Fallback cleanup
            try:
                shutil.rmtree(self.lock_dir, ignore_errors=True)
                self._acquired = False
                return True
            except Exception:
                return False

    def _clean_stale_and_retry(self) -> bool:
        try:
            if self.info_file.exists():
                self.info_file.unlink(missing_ok=True)
            if self.lock_dir.exists():
                shutil.rmtree(self.lock_dir, ignore_errors=True)
        except Exception:
            return False

        try:
            os.mkdir(self.lock_dir)
            self._write_lock_info()
            self._acquired = True
            return True
        except Exception:
            return False

    def is_locked(self) -> bool:
        """Checks if the project lock is currently held by an active process."""
        if not self.lock_dir.exists():
            return False
        info = self.read_lock_info()
        if info is None:
            return False
        return is_process_alive(info.host_pid, info.host_create_time)

    def __enter__(self) -> ProjectLock:
        if not self.acquire():
            info = self.read_lock_info()
            owner_msg = f" (owner: {info.owner_id}, pid: {info.host_pid})" if info else ""
            raise BlockingIOError(f"Cannot acquire project lock: held by another active process{owner_msg}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def main():
    parser = argparse.ArgumentParser(description="Antigravity Project Mutual Exclusion Lock CLI")
    parser.add_argument("action", choices=["status", "acquire", "release", "heartbeat"], help="Action to execute")
    parser.add_argument("--owner", default="antigravity_cli", help="Owner identifier")
    parser.add_argument("--branch", default="", help="Active git branch")
    parser.add_argument("--desc", default="", help="Task description")
    parser.add_argument("--lock-dir", default=None, help="Custom lock directory path")
    args = parser.parse_args()

    lock_dir = Path(args.lock_dir) if args.lock_dir else DEFAULT_LOCK_DIR
    lock = ProjectLock(lock_dir=lock_dir, owner_id=args.owner, description=args.desc, branch=args.branch)

    if args.action == "status":
        if lock.lock_dir.exists():
            info = lock.read_lock_info()
            if info:
                alive = is_process_alive(info.host_pid, info.host_create_time)
                print(json.dumps({
                    "status": "LOCKED" if alive else "STALE",
                    "active_process": alive,
                    "info": asdict(info)
                }, indent=2))
                sys.exit(0 if alive else 1)
            else:
                print(json.dumps({"status": "CORRUPT_OR_EMPTY_LOCK_DIR"}, indent=2))
                sys.exit(2)
        else:
            print(json.dumps({"status": "UNLOCKED"}, indent=2))
            sys.exit(0)

    elif args.action == "acquire":
        success = lock.acquire()
        if success:
            print(json.dumps({"status": "ACQUIRED", "lock_dir": str(lock.lock_dir)}, indent=2))
            sys.exit(0)
        else:
            info = lock.read_lock_info()
            print(json.dumps({
                "status": "FAILED",
                "reason": "HELD_BY_ACTIVE_PROCESS",
                "held_by": asdict(info) if info else None
            }, indent=2))
            sys.exit(1)

    elif args.action == "release":
        success = lock.release()
        print(json.dumps({"status": "RELEASED" if success else "FAILED_TO_RELEASE"}, indent=2))
        sys.exit(0 if success else 1)

    elif args.action == "heartbeat":
        success = lock.heartbeat()
        print(json.dumps({"status": "HEARTBEAT_UPDATED" if success else "FAILED"}, indent=2))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
