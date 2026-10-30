"""Hourly Coordination & Task Runner with Real Mutual Exclusion (CX-0003 / CX-0008 / CX-0010).

Guarantees:
1. Acquires `ProjectLock` (OS-level kernel descriptor lock + metadata) BEFORE inspecting
   or mutating state, and holds the lock across the entire check + authorized task execution.
2. Executes `git fetch origin` without modifying local working tree files.
3. Reads remote messages directly from `origin/main` via `git ls-tree` and `git show`
   (never relying on local checkout files that `git fetch` does not update).
4. Prevents duplicate execution of completed `task_id`s and executes any authorized command
   (`--task-cmd`) inside the critical section while pulsing lock heartbeats.
5. Logs start/end timestamps, observed HEADs, unread remote messages, and execution result
   to an append-only JSONL audit log.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.coordination_lock import DEFAULT_LOCK_DIR, ProjectLock, is_process_alive

DEFAULT_LOG_FILE = REPO_ROOT / "results" / "intensive_20261030" / "coordination_runner_log.jsonl"
DEFAULT_STATE_FILE = REPO_ROOT / "coordination" / "CHECKPOINT_STATE.json"


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def run_git(args: List[str], cwd: Path = REPO_ROOT, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def list_remote_codex_messages(ref: str = "origin/main", cwd: Path = REPO_ROOT) -> List[Dict[str, str]]:
    """Lists CX-*.md messages directly from a git ref (e.g. origin/main) via git ls-tree & git show."""
    res = run_git(["ls-tree", ref, "coordination/codex_to_antigravity/"], cwd=cwd)
    if res.returncode != 0:
        return []

    messages: List[Dict[str, str]] = []
    for line in res.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        blob_sha = parts[2]
        rel_path = parts[3]
        name = Path(rel_path).name
        if not (name.startswith("CX-") and name.endswith(".md")):
            continue

        msg_id = name.split("-")[0] + "-" + name.split("-")[1].replace(".md", "")
        show_res = run_git(["show", f"{ref}:{rel_path}"], cwd=cwd)
        subject_type = ""
        in_reply_to = ""
        supersedes = ""
        if show_res.returncode == 0:
            for mline in show_res.stdout.splitlines()[:20]:
                if mline.startswith("id:"):
                    msg_id = mline.split(":", 1)[1].strip()
                elif mline.startswith("type:"):
                    subject_type = mline.split(":", 1)[1].strip()
                elif mline.startswith("in_reply_to:"):
                    in_reply_to = mline.split(":", 1)[1].strip()
                elif mline.startswith("supersedes:"):
                    supersedes = mline.split(":", 1)[1].strip()

        messages.append(
            {
                "id": msg_id,
                "file": name,
                "path": rel_path,
                "blob_sha": blob_sha,
                "type": subject_type,
                "in_reply_to": in_reply_to,
                "supersedes": supersedes,
            }
        )

    def _msg_sort_key(item: Dict[str, str]) -> int:
        m = re.search(r"CX-(\d+)", item["id"])
        return int(m.group(1)) if m else 0

    messages.sort(key=_msg_sort_key)
    return messages


def _msg_num(msg_id: str) -> int:
    m = re.search(r"CX-(\d+)", msg_id or "")
    return int(m.group(1)) if m else 0


def append_runner_log(log_file: Path, record: Dict) -> None:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def has_completed_task(log_file: Path, task_id: str) -> bool:
    if not task_id or not log_file.exists():
        return False
    try:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("task_id") == task_id
                and row.get("status") == "COMPLETED"
                and row.get("task_exit_code") == 0
            ):
                return True
    except Exception:
        pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronized coordination check and task runner")
    parser.add_argument("--lock-dir", default=None, help="Custom lock directory path")
    parser.add_argument("--owner", default="antigravity_hourly_runner", help="Lock owner identifier")
    parser.add_argument("--branch", default="antigravity/overnight-recovery-coordination-20261030")
    parser.add_argument("--remote-ref", default="origin/main", help="Git ref to inspect for Codex messages")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip git fetch origin (for offline tests)")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Path to CHECKPOINT_STATE.json")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Path to runner JSONL log")
    parser.add_argument("--task-id", default="", help="Unique task identifier to prevent duplicate execution")
    parser.add_argument("--task-cmd", default="", help="Shell/command to execute inside the critical section")
    args = parser.parse_args()

    lock_dir = Path(args.lock_dir) if args.lock_dir else DEFAULT_LOCK_DIR
    log_file = Path(args.log_file)
    state_file = Path(args.state_file)
    started_utc = _now_utc()
    started_local = _now_local()

    lock = ProjectLock(
        lock_dir=lock_dir,
        owner_id=args.owner,
        description=f"Hourly coordination check / task={args.task_id or 'check_only'}",
        branch=args.branch,
    )

    if not lock.acquire():
        info = lock.read_lock_info()
        record = {
            "status": "SKIPPED_LOCK_HELD",
            "started_at_utc": started_utc,
            "started_at_local": started_local,
            "finished_at_utc": _now_utc(),
            "owner_attempting": args.owner,
            "task_id": args.task_id,
            "held_by": {
                "owner_id": info.owner_id if info else None,
                "host_pid": info.host_pid if info else None,
                "started_at_utc": info.started_at_utc if info else None,
                "heartbeat_utc": info.heartbeat_utc if info else None,
                "alive": is_process_alive(info.host_pid, info.host_create_time) if info else None,
            },
        }
        append_runner_log(log_file, record)
        print(json.dumps(record, indent=2))
        return 0

    try:
        fetch_ok = True
        if not args.skip_fetch:
            fetch_res = run_git(["fetch", "origin"], timeout=30)
            fetch_ok = fetch_res.returncode == 0

        local_head_res = run_git(["rev-parse", "HEAD"])
        local_head = local_head_res.stdout.strip() if local_head_res.returncode == 0 else ""

        remote_head_res = run_git(["rev-parse", args.remote_ref])
        remote_head = remote_head_res.stdout.strip() if remote_head_res.returncode == 0 else ""

        last_processed = ""
        if state_file.exists():
            try:
                state_data = json.loads(state_file.read_text(encoding="utf-8"))
                last_processed = state_data.get("last_processed_codex_message", "")
            except Exception:
                pass

        remote_messages = list_remote_codex_messages(ref=args.remote_ref)
        last_num = _msg_num(last_processed)
        pending_messages = [m for m in remote_messages if _msg_num(m["id"]) > last_num]

        task_exit_code: Optional[int] = None
        task_status = "NO_TASK"

        if args.task_cmd:
            if args.task_id and has_completed_task(log_file, args.task_id):
                task_status = "SKIPPED_DUPLICATE_TASK"
                task_exit_code = 0
            else:
                lock.heartbeat()
                proc = subprocess.Popen(
                    args.task_cmd,
                    cwd=str(REPO_ROOT),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                while proc.poll() is None:
                    lock.heartbeat()
                    time.sleep(0.2)
                task_exit_code = proc.returncode
                task_status = "COMPLETED" if task_exit_code == 0 else "FAILED"

        record = {
            "status": task_status if args.task_cmd else "CHECK_COMPLETED",
            "started_at_utc": started_utc,
            "started_at_local": started_local,
            "finished_at_utc": _now_utc(),
            "finished_at_local": _now_local(),
            "owner": args.owner,
            "fetch_ok": fetch_ok,
            "local_head": local_head,
            "remote_ref": args.remote_ref,
            "remote_head": remote_head,
            "last_processed_codex_message": last_processed,
            "remote_codex_messages_count": len(remote_messages),
            "latest_remote_codex_message": remote_messages[-1]["id"] if remote_messages else None,
            "pending_codex_messages": [m["id"] for m in pending_messages],
            "task_id": args.task_id,
            "task_cmd": args.task_cmd,
            "task_exit_code": task_exit_code,
        }
        append_runner_log(log_file, record)
        print(json.dumps(record, indent=2))
        return 0 if (task_exit_code is None or task_exit_code == 0) else task_exit_code
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
