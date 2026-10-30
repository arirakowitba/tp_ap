"""Atomic Project-Level Mutual Exclusion Lock (CX-0003 / CX-0004 / CX-0008 / CX-0010).

Implements mutual exclusion held across the entire critical section using:
1. An OS-level kernel file lock (`msvcrt.locking` on Windows / `fcntl.flock` on POSIX)
   kept open for the entire lifetime of the lock acquisition (automatically released
   by the OS kernel if the process crashes or is killed).
2. In-process instance token tracking so two `ProjectLock` instances in the same
   process cannot collide or release each other's lock.
3. Atomic directory + atomic temp-file replacement for `lock_info.json`.
4. Safe stale-lock recovery that NEVER deletes a newly created lock directory merely
   because metadata (`lock_info.json`) has not finished writing yet.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import stat
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

if os.name == "nt":
    import msvcrt
else:
    import fcntl

try:
    import psutil
except ImportError:
    psutil = None

DEFAULT_LOCK_DIR = Path(__file__).resolve().parent.parent / "coordination" / ".project_lock"

_PROCESS_LOCK_GUARD = threading.Lock()
_ACTIVE_PROCESS_CLAIMS: Dict[str, str] = {}


def _force_rmdir(path: Path) -> None:
    """Removes a directory tree, clearing read-only attributes if needed (e.g. OneDrive on Windows)."""
    if not path.exists():
        return

    def _handle_remove_readonly(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    except Exception:
        pass


def _now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _now_iso_local() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def is_process_alive(pid: int, expected_create_time: Optional[float] = None) -> bool:
    """Checks whether the process with PID is currently alive and matches create_time."""
    if pid <= 0:
        return False
    if psutil is None:
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

        if expected_create_time is not None and expected_create_time > 0:
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
    instance_token: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> LockInfo:
        data = json.loads(text)
        allowed = {
            "owner_id",
            "host_pid",
            "host_create_time",
            "started_at_utc",
            "started_at_local",
            "heartbeat_utc",
            "heartbeat_local",
            "branch",
            "description",
            "instance_token",
        }
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)


class ProjectLock:
    """Atomic project mutual exclusion lock manager with OS kernel descriptor locking."""

    def __init__(
        self,
        lock_dir: Optional[Path] = None,
        owner_id: str = "antigravity",
        description: str = "",
        branch: str = "",
        empty_dir_stale_seconds: float = 10.0,
    ):
        self.lock_dir = Path(lock_dir) if lock_dir else DEFAULT_LOCK_DIR
        self.info_file = self.lock_dir / "lock_info.json"
        self.os_lock_file = self.lock_dir.parent / f"{self.lock_dir.name}.oslock"
        self.owner_id = owner_id
        self.description = description
        self.branch = branch
        self.empty_dir_stale_seconds = empty_dir_stale_seconds
        self.instance_token = uuid.uuid4().hex
        self._acquired = False
        self._os_lock_fd: Optional[int] = None

    def _lock_key(self) -> str:
        try:
            return str(self.lock_dir.resolve()).lower()
        except Exception:
            return str(self.lock_dir).lower()

    def _acquire_os_lock(self) -> bool:
        """Acquires both the in-process claim and an OS kernel non-blocking file lock."""
        key = self._lock_key()
        with _PROCESS_LOCK_GUARD:
            existing_token = _ACTIVE_PROCESS_CLAIMS.get(key)
            if existing_token is not None and existing_token != self.instance_token:
                return False
            _ACTIVE_PROCESS_CLAIMS[key] = self.instance_token

        fd: Optional[int] = None
        try:
            self.os_lock_file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.os_lock_file), os.O_CREAT | os.O_RDWR, 0o666)
            if os.name == "nt":
                # Ensure at least 1 byte exists and lock byte 0 non-blocking
                try:
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\x00")
                except Exception:
                    pass
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._os_lock_fd = fd
            return True
        except (OSError, IOError, PermissionError, BlockingIOError):
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            with _PROCESS_LOCK_GUARD:
                if _ACTIVE_PROCESS_CLAIMS.get(key) == self.instance_token:
                    _ACTIVE_PROCESS_CLAIMS.pop(key, None)
            return False

    def _release_os_lock(self) -> None:
        """Releases the OS kernel file lock and in-process claim held by this instance."""
        fd = self._os_lock_fd
        self._os_lock_fd = None
        if fd is not None:
            try:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass
        key = self._lock_key()
        with _PROCESS_LOCK_GUARD:
            if _ACTIVE_PROCESS_CLAIMS.get(key) == self.instance_token:
                _ACTIVE_PROCESS_CLAIMS.pop(key, None)

    def acquire(self, retry_stale: bool = True) -> bool:
        """Attempts atomic acquisition of the project lock.

        Holds the OS kernel file lock across the entire critical section.
        Never deletes a lock directory simply because metadata is not yet present.
        """
        if self._acquired:
            return True

        if not self._acquire_os_lock():
            return False

        try:
            os.mkdir(self.lock_dir)
            self._write_lock_info()
            self._acquired = True
            return True
        except FileExistsError:
            # We hold the OS lock, yet lock_dir exists on disk (e.g. previous crash or manual fixture).
            existing_info = self.read_lock_info()
            if existing_info is None:
                # Never delete a freshly created directory whose metadata may still be in flight.
                if not retry_stale or not self._is_empty_dir_stale():
                    self._release_os_lock()
                    return False
                if self._clean_stale_and_retry():
                    return True
                self._release_os_lock()
                return False

            # Check if existing host process is still alive
            if is_process_alive(existing_info.host_pid, existing_info.host_create_time):
                self._release_os_lock()
                return False

            # Recorded process is dead (stale lock after crash)
            if retry_stale and self._clean_stale_and_retry():
                return True

            self._release_os_lock()
            return False
        except Exception:
            self._release_os_lock()
            return False

    def _is_empty_dir_stale(self) -> bool:
        """Returns True only if lock_dir has no metadata AND is older than empty_dir_stale_seconds."""
        try:
            st = self.lock_dir.stat()
            age = max(0.0, time.time() - st.st_mtime)
            return age >= self.empty_dir_stale_seconds
        except Exception:
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
            instance_token=self.instance_token,
        )
        tmp_file = self.lock_dir / f".lock_info.{self.instance_token}.tmp"
        tmp_file.write_text(info.to_json(), encoding="utf-8")
        os.replace(tmp_file, self.info_file)

    def read_lock_info(self) -> Optional[LockInfo]:
        """Reads LockInfo if info_file exists and is valid."""
        if not self.info_file.exists():
            return None
        try:
            return LockInfo.from_json(self.info_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def heartbeat(self) -> bool:
        """Updates the heartbeat timestamp if this exact instance owns the lock."""
        if not self._acquired or self._os_lock_fd is None:
            return False
        info = self.read_lock_info()
        if (
            info is None
            or info.host_pid != os.getpid()
            or info.owner_id != self.owner_id
            or (info.instance_token and info.instance_token != self.instance_token)
        ):
            return False
        info.heartbeat_utc = _now_iso_utc()
        info.heartbeat_local = _now_iso_local()
        tmp_file = self.lock_dir / f".lock_info.{self.instance_token}.tmp"
        tmp_file.write_text(info.to_json(), encoding="utf-8")
        os.replace(tmp_file, self.info_file)
        return True

    def release(self) -> bool:
        """Releases the lock ONLY if acquired by this exact instance."""
        if not self._acquired:
            return False

        info = self.read_lock_info()
        if info is not None:
            if info.host_pid != os.getpid() or info.owner_id != self.owner_id:
                return False
            if info.instance_token and info.instance_token != self.instance_token:
                return False

        try:
            if self.info_file.exists():
                try:
                    os.chmod(self.info_file, stat.S_IWRITE)
                except Exception:
                    pass
                self.info_file.unlink(missing_ok=True)
            if self.lock_dir.exists():
                os.rmdir(self.lock_dir)
            self._acquired = False
            self._release_os_lock()
            return True
        except Exception:
            try:
                _force_rmdir(self.lock_dir)
                self._acquired = False
                self._release_os_lock()
                return True
            except Exception:
                return False

    def _clean_stale_and_retry(self) -> bool:
        """Cleans a verified stale lock directory while holding _os_lock_fd and retries."""
        if self._os_lock_fd is None:
            return False
        existing_info = self.read_lock_info()
        if existing_info is None and not self._is_empty_dir_stale():
            return False
        if existing_info is not None and is_process_alive(
            existing_info.host_pid, existing_info.host_create_time
        ):
            return False

        try:
            if self.info_file.exists():
                try:
                    os.chmod(self.info_file, stat.S_IWRITE)
                except Exception:
                    pass
                self.info_file.unlink(missing_ok=True)
            if self.lock_dir.exists():
                _force_rmdir(self.lock_dir)
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
            return not self._is_empty_dir_stale()
        return is_process_alive(info.host_pid, info.host_create_time)

    def __enter__(self) -> ProjectLock:
        if not self.acquire():
            info = self.read_lock_info()
            owner_msg = f" (owner: {info.owner_id}, pid: {info.host_pid})" if info else ""
            raise BlockingIOError(
                f"Cannot acquire project lock: held by another active process{owner_msg}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def main():
    parser = argparse.ArgumentParser(description="Antigravity Project Mutual Exclusion Lock CLI")
    parser.add_argument(
        "action", choices=["status", "acquire", "release", "heartbeat"], help="Action to execute"
    )
    parser.add_argument("--owner", default="antigravity_cli", help="Owner identifier")
    parser.add_argument("--branch", default="", help="Active git branch")
    parser.add_argument("--desc", default="", help="Task description")
    parser.add_argument("--lock-dir", default=None, help="Custom lock directory path")
    args = parser.parse_args()

    lock_dir = Path(args.lock_dir) if args.lock_dir else DEFAULT_LOCK_DIR
    lock = ProjectLock(
        lock_dir=lock_dir, owner_id=args.owner, description=args.desc, branch=args.branch
    )

    if args.action == "status":
        if lock.lock_dir.exists():
            info = lock.read_lock_info()
            if info:
                alive = is_process_alive(info.host_pid, info.host_create_time)
                print(
                    json.dumps(
                        {
                            "status": "LOCKED" if alive else "STALE",
                            "active_process": alive,
                            "info": asdict(info),
                        },
                        indent=2,
                    )
                )
                sys.exit(0 if alive else 1)
            else:
                print(json.dumps({"status": "INITIALIZING_OR_EMPTY_LOCK_DIR"}, indent=2))
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
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "reason": "HELD_BY_ACTIVE_PROCESS",
                        "held_by": asdict(info) if info else None,
                    },
                    indent=2,
                )
            )
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
