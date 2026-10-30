"""Hourly Coordination Runner & Project Lock Integration (CX-0003, CX-0004, CX-0006).

Provides an atomic entry point for automated or manual coordination runs.
Integrates ProjectLock to guarantee strict mutual exclusion across all sessions
and background runners.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordination_lock import ProjectLock


def run_coordination_cycle(owner: str = "antigravity_hourly", verbose: bool = True) -> int:
    """Executes a single coordination cycle protected by ProjectLock."""
    lock = ProjectLock(owner_id=owner, description="Scheduled hourly coordination check")
    
    if verbose:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to acquire project lock for {owner}...")

    try:
        with lock:
            if verbose:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Project lock acquired successfully.")

            # 1. Fetch origin to inspect incoming messages
            if verbose:
                print("Fetching origin/main...")
            fetch_res = subprocess.run(["git", "fetch", "origin"], cwd=str(ROOT), capture_output=True, text=True)
            if fetch_res.returncode != 0:
                print(f"Warning: git fetch failed: {fetch_res.stderr}")

            # 2. Check origin/main HEAD
            rev_res = subprocess.run(["git", "rev-parse", "origin/main"], cwd=str(ROOT), capture_output=True, text=True)
            origin_head = rev_res.stdout.strip() if rev_res.returncode == 0 else "UNKNOWN"
            if verbose:
                print(f"Observed origin/main HEAD: {origin_head}")

            # 3. List recent codex messages
            codex_dir = ROOT / "coordination" / "codex_to_antigravity"
            codex_msgs = sorted([p.name for p in codex_dir.glob("CX-*.md")])
            if verbose:
                print(f"Local codex messages present: {codex_msgs}")

            # 4. Heartbeat update
            lock.heartbeat()

            if verbose:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Coordination cycle complete. Releasing lock.")
            return 0

    except BlockingIOError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lock collision: {e}")
        return 1
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Coordination runner error: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Run coordination tick under atomic ProjectLock")
    parser.add_argument("--owner", default="antigravity_hourly", help="Owner ID for lock")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    sys.exit(run_coordination_cycle(owner=args.owner, verbose=not args.quiet))


if __name__ == "__main__":
    main()
