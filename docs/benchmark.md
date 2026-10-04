# Benchmark harness

The evaluator scores a `plan.json` against a ground-truth file. It does not reconstruct, and it does not fill in a measurement the plan left out.

Three different things must stay separate.

| | What it is | What it is not |
| --- | --- | --- |
| Evaluator correctness | The tests in `tests/test_benchmark.py` | A claim that a capture is accurate |
| Synthetic evaluator fixtures | `benchmarks/fixtures/` | Laser or tape ground truth |
| Physical benchmark | A case whose ground truth has `provenance: laser_tape` | Anything in this repo today |

Physical gates remain **blocked** until a laser or tape survey is available. A synthetic fixture can make `pass: true` inside a metric block. The case still reports `assessment_gate: blocked`.

## Layout

```
benchmarks/schema/ground_truth.schema.json
benchmarks/fixtures/<case>/ground_truth.json
benchmarks/fixtures/<case>/prediction.json
benchmarks/fixtures/<case>/prediction_repeat.json   # only for repeatability
benchmarks/reports/<case>/report.json
benchmarks/reports/<case>/report.txt
```

`prediction.json` is a `plan.json`. There is no second prediction format. Point `--prediction` at `runs/.../plan.json` directly.

```bash
python -m cozmo_scan.benchmark_eval --prediction runs/single_room_lidar/plan.json --case-id lidar_sample --out benchmarks/reports/lidar_sample
python -m cozmo_scan.benchmark_eval --case benchmarks/fixtures/exact_match --out benchmarks/reports/exact_match
```

A second walk is `--repeat`. A plan built with drift correction off is `--drift-off`. If those files are absent, the metric is `unavailable` and any drift record already inside the plan is copied through as evidence.

## Ground truth

`ground_truth.json` holds the survey, not the prediction. Lengths are metres.

- `rooms[].room_id`
- `rooms[].walls[]` with `length_m` and, when ids will not match, `p0_m` / `p1_m`
- `rooms[].openings[]` with `wall_id` and `width_m`
- `rooms[].ceiling_height_m`
- `footprint_m2`
- `adjacencies[]` with `room_a` and `room_b`
- `overlap_m2_max`
- `calibration.scale`: `measured`, `estimated`, or `unavailable`
- `room_map`: prediction room id to ground-truth room id, for a capture whose ids are not the survey ids
- `provenance`: `synthetic_evaluator` or `laser_tape`

## Matching

Rooms match on `room_id`, after `room_map` is applied. An id that matches nothing is listed as unmatched. Walls from that room are not guessed onto another room.

Walls match on `wall_id` / `id` when both sides have that id. Otherwise a ground-truth edge is paired with the nearest predicted edge of similar orientation (within 20 degrees) and the closest midpoint. Ties break on wall id. A ground-truth wall with no partner is a miss. A predicted wall with no partner is a phantom.

Openings match on `opening_id` / `id` first. Remaining openings on the same wall are paired in order of width, then id. A leftover ground-truth opening is a miss. A leftover prediction is a phantom. Both count as failures.

Repeatability matches walls between two plans of the same room. The second length is `compared_m`. It is not called ground truth.

## Thresholds

Named in `cozmo_scan/benchmark_eval.py`. A test may pass a different `Thresholds` object. The defaults are the assessment values.

| Name | Value | Use |
| --- | --- | --- |
| `OPENING_MAX_ERROR_M` | 0.02 | one opening passes within 2 cm |
| `OPENING_PASS_RATE` | 0.85 | case gate, misses and phantoms included |
| `CEILING_MAX_ERROR_M` | 0.015 | ceiling within 1.5 cm |
| `CEILING_REPEAT_SPREAD_M` | 0.01 | two ceilings of the same room |
| `REPEATABILITY_ABS_M` | 0.01 | two captures, absolute |
| `REPEATABILITY_REL` | 0.005 | two captures, or the absolute rule |
| `PHOTO_FOOTPRINT_REL` | 0.08 | photo footprint |
| `PHOTO_WALL_REL` | 0.08 | photo wall length |
| `VIDEO_WALL_REL` | 0.03 | video wall length |

LiDAR wall length has no percentage in that list. The harness still reports absolute and relative error. `pass` stays null unless `wall_abs_m` is set. That is not a hidden LiDAR gate.

Relative error is `|predicted - ground_truth| / |ground_truth|`. Repeatability uses `|a - b| / max(|a|, |b|)`, and passes when the absolute difference is at most 1 cm or that ratio is at most 0.5%.

## Status values

`evaluated` means both numbers were present and the arithmetic ran. `unavailable` means the metric applies but a file or a field is missing. `not_applicable` means the metric is for another tier, or there was nothing to score (no openings on either side, no ceiling in the ground truth). `blocked` is reserved for the assessment gate until the ground truth says `laser_tape`.

Calibration is `estimated` only when the plan says the scale is an estimate. A photo fixture that stored metres in a JSON file is `unavailable`: those metres are not a tape. The 1.40 m camera-height prior is never treated as ground truth. LiDAR and video plans that carry no calibration record stay `unavailable`.

Drift copies the plan's method, tilt, chunk offsets, loop note, and anchor-on / anchor-off footprint. A wall-by-wall on/off comparison runs only when a drift-off plan is passed in.
