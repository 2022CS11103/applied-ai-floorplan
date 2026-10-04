"""Regenerate every number cited in docs/benchmark_report.md and docs/fix_loop.md.

The sample folders have to be on disk. They are not in git.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

COMMANDS = [
    [PY, "run.py", "self-check"],
    [PY, "run.py", "run", "--capture", "single_room/c00a170fe1", "--tier", "lidar",
     "--endpoints", "intersections", "--out", "runs/single_room_lidar"],
    [PY, "run.py", "run", "--capture", "single_room/c00a170fe1", "--tier", "lidar",
     "--endpoints", "observed", "--out", "runs/single_room_before"],
    [PY, "run.py", "run", "--capture", "single_scan_with_ceiling/c7d28f72c6", "--tier", "lidar",
     "--out", "runs/ceiling_lidar"],
    [PY, "run.py", "run", "--capture", "single_room/c00a170fe1", "--tier", "video",
     "--out", "runs/single_room_video"],
    [PY, "run.py", "repeat", "--capture", "single_room/c00a170fe1",
     "--endpoints", "observed", "--out", "runs/repeat_before"],
    [PY, "run.py", "repeat", "--capture", "single_room/c00a170fe1",
     "--endpoints", "intersections", "--out", "runs/repeat_after"],
]


def main() -> None:
    for cmd in COMMANDS:
        print(" ".join(cmd[1:]))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
