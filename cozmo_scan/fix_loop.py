"""Regenerate a before/after pair and write the difference.

An even/odd split, or two endpoint modes of one walk, is a proxy. It is
labelled proxy_experiment and it is not the physical repeatability gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark_eval import REPEATABILITY_ABS_M


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _worst_wall(plan: dict | None) -> float | None:
    if not plan:
        return None
    lengths = []
    for room in (plan.get("property") or {}).get("rooms") or []:
        for wall in room.get("walls") or []:
            value = wall.get("length_m")
            if isinstance(value, dict):
                value = value.get("value")
            if value is not None:
                lengths.append(float(value))
    return None if not lengths else max(lengths)


def run_side(capture: Path, out_dir: Path, endpoint_mode: str, tier: str = "lidar") -> dict:
    """Run one capture. The benchmark file says this is not two physical walks."""
    from .pipeline import run_one

    out_dir = Path(out_dir)
    plan = run_one(Path(capture), tier, out_dir, endpoint_mode=endpoint_mode)
    benchmark = {
        "status": "BLOCKED",
        "reason": "one capture cannot satisfy the two-capture repeatability gate",
        "proxy_experiment": True,
        "endpoint_mode": endpoint_mode,
        "threshold_m": REPEATABILITY_ABS_M,
        "rooms": len((plan.get("property") or {}).get("rooms") or []),
    }
    _write(out_dir / "benchmark.json", benchmark)
    return {"plan": plan, "benchmark": benchmark}


def before(capture: Path, out_dir: Path, tier: str = "lidar") -> dict:
    """The shipped fix compares observed endpoints with corner intersections."""
    return run_side(capture, Path(out_dir) / "before", "observed", tier=tier)


def after(capture: Path, out_dir: Path, tier: str = "lidar") -> dict:
    return run_side(capture, Path(out_dir) / "after", "intersections", tier=tier)


def compare(before_dir: Path, after_dir: Path, out_dir: Path | None = None) -> dict:
    """Read two already written sides. Does not invent a measurement."""
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)
    destination = Path(out_dir) if out_dir else before_dir.parent
    destination.mkdir(parents=True, exist_ok=True)

    def load(folder: Path, name: str) -> dict | None:
        path = folder / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    plan_before = load(before_dir, "plan.json")
    plan_after = load(after_dir, "plan.json")
    bench_before = load(before_dir, "benchmark.json") or {}
    bench_after = load(after_dir, "benchmark.json") or {}
    missing = []
    if plan_before is None:
        missing.append("before/plan.json")
    if plan_after is None:
        missing.append("after/plan.json")
    before_value = _worst_wall(plan_before)
    after_value = _worst_wall(plan_after)
    delta = None if before_value is None or after_value is None else round(after_value - before_value, 4)
    proxy = bool(bench_before.get("proxy_experiment", True) or bench_after.get("proxy_experiment", True))
    if missing:
        status_before = "BLOCKED"
        status_after = "BLOCKED"
    else:
        status_before = bench_before.get("status") or "BLOCKED"
        status_after = bench_after.get("status") or "BLOCKED"
    diff = {
        "gate": "wall_repeatability",
        "before": before_value,
        "after": after_value,
        "delta": delta,
        "threshold": REPEATABILITY_ABS_M,
        "status_before": status_before,
        "status_after": status_after,
        "proxy_experiment": proxy,
        "missing": missing,
    }
    _write(destination / "diff.json", diff)
    predicted = "The intersection endpoints should shorten a wall that previously ran past the corner."
    actual = "No after plan was produced." if plan_after is None else f"Longest wall after the change is {after_value} m."
    missed = ""
    if before_value is not None and after_value is not None and abs(delta or 0) < 1e-6:
        missed = "The predicted change did not move the longest wall. The hypothesis does not explain this capture."
    elif proxy:
        missed = "This pair is one capture viewed two ways. It does not close the physical repeatability gate."
    report = "\n".join(
        [
            "# Fix loop",
            "",
            f"1. Worst gate: {diff['gate']}.",
            f"2. Failing number: before {before_value} m, threshold {REPEATABILITY_ABS_M} m.",
            "3. Root-cause hypothesis: observed wall ends stop at the last return and can overshoot a corner.",
            "4. Evidence: before/plan.json and after/plan.json from the same capture.",
            "5. Intended fix: place the wall end at the intersection of the two wall lines.",
            f"6. Predicted result: {predicted}",
            f"7. Actual after result: {actual}",
            f"8. Honest explanation: {missed or 'The after plan was written. Physical repeatability still needs a second walk.'}",
            "",
            f"proxy_experiment: {str(proxy).lower()}",
            "",
        ]
    )
    (destination / "report.md").write_text(report, encoding="utf-8")
    return diff
