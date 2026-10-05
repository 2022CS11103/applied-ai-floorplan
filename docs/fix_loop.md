# Fix loop

Capture: `single_room/c00a170fe1`, LiDAR tier.
Proxy, not two physical walks: even frames against odd frames of the same walk. A second walk was not possible. The proxy is labelled `proxy_experiment: true` in `repeatability.json`. It is not the official repeatability gate. That gate is `compare_two_captures` in `cozmo_scan/repeatability.py`, and it stays BLOCKED until capture B exists.

Regenerate the write-up from two already written plans:

```bash
python run.py fix-loop before --capture <lidar-folder> --out runs/fix_loop
python run.py fix-loop after --capture <lidar-folder> --out runs/fix_loop
python run.py fix-loop compare --before runs/fix_loop/before --after runs/fix_loop/after
```

`compare` writes `diff.json` and `report.md` with the worst gate, the failing number, the hypothesis, the evidence, the intended fix, the predicted result, the actual after result, and an honest note when the prediction missed. One capture keeps `proxy_experiment: true`.

## 1. Worst gate

Repeatability. Two passes of the same room should agree within 1 cm or 0.5% on a wall, whichever is larger.

Before the fix, the worst matched wall moved by **0.05 m**. Its tolerance was **0.0177 m**. One of the two matched walls passed, one failed. Gate failed.

## 2. Root cause

The wall length was the stretch of surface those frames happened to see.

Evidence, from `runs/repeat_before/repeatability.json`:

| Pass | Length of the same wall |
| --- | --- |
| even frames | 3.55 m |
| odd frames | 3.60 m |

The midpoints were 11 cm apart, so it is the same wall. The 5 cm gap is the difference in how far along that wall each subset got returns, not a change in the room. The full before plan (`runs/single_room_before`) shows the same mistake at capture scale: observed runs of 4.65 m on a room whose other side is 2.40 m.

## 3. The fix, and the number it was aimed at

Report the distance between the two cross-walls that close the room. Both subsets still see those cross-walls even when they see different portions of the long wall. Predicted worst delta after the fix: under 1 cm.

Shipped as the default `--endpoints intersections`. The old behaviour remains `--endpoints observed`, so the before run is regenerable.

## 4. After

`runs/repeat_after/repeatability.json`:

| | Before (`observed`) | After (`intersections`) |
| --- | --- | --- |
| Walls compared | 2 | 2 |
| Inside the gate | 1 | 2 |
| Worst absolute delta | 0.050 m | 0.0052 m |
| Gate | fail | pass on the walls that matched |

0.52 cm is inside the 1.2 cm tolerance on a 2.40 m wall. The prediction (under 1 cm) was in the right direction and inside the gate.

What did not get fixed: even frames still close 3 rooms and odd frames close 2. The split agrees on the walls it can match. It does not agree on how many rooms the furniture ridges produce. That is a coverage and clutter problem, and it is still open. The report says so rather than calling the whole repeatability row solved for a second physical capture.

## Regenerate

```bash
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_before --endpoints observed
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_after --endpoints intersections
python run.py run --capture single_room/c00a170fe1 --tier lidar --endpoints observed --out runs/single_room_before
python run.py run --capture single_room/c00a170fe1 --tier lidar --endpoints intersections --out runs/single_room_lidar
```

The pass in the table above is the even/odd proxy on one company walk. It is not the PDF repeatability gate. Two ARKit walks of visit 421337 still differ by 9.7 cm on the long wall.

## Physical repeatability, not closed

Gate: two walks of the same room, absolute difference ≤ 1 cm or relative difference ≤ 0.5%.

Failing number: ARKitScenes visit 421337, long wall 9.7 cm (2.4%) between the two Apple walks. Short wall 1.5 cm (0.41%). Ceiling spread 1.1 cm.

Root cause: these are two real walks, not an even/odd split of one walk. The even/odd fix above does not move this pair onto the bar.

Evidence: `benchmarks/external/arkitscenes_visit_421337/repeatability.json`.

Shipped fix: none. No threshold was moved, and the wall length was not pulled toward the other walk.

Before: 9.7 cm. After: 9.7 cm. Reproduction: `python -m datasets.arkitscenes_reference`.

## Opening width, not closed

Worst remaining physical gate: opening width ≤ 2 cm on ≥ 85%.

Failing number: self-check door 0.863 m against 0.80 m (6.3 cm). One fused grid of the 62 synthetic openings is 39/62. There is no tape, so the PDF gate stays BLOCKED rather than PASS.

Root cause, from the synthetic failure split: 19 openings leave the jamb in a gap between returns, 4 drop a wall column, and the old 3-hit rule missed 5 sparse bins. A second camera position supplies the missing return. A sparse bin is now a separate low-confidence candidate. The shipped LiDAR command still voxel-fuses before the opening detector, so those synthetic frame shifts are not the company-scan result.

Evidence: `tests/test_opening_multiframe.py` and `docs/benchmark_report.md`. Before: 34/62 on one synthetic frame. After the sparse path: 39/62 on one frame. Five synthetic frames: 62/62. That synthetic schedule is not a real-gate fix. Reproduction: `python -m pytest tests/test_opening_multiframe.py`.

## Assessment command, not a physical pass

`python run.py assessment --manifest <path>` validates one property and writes `runs/assessment/<id>/`. The example manifest has no captures and no tape file, so every gate stays BLOCKED. A video room that does not close is DEGRADED. Neither state is a pass, and the synthetic five-frame opening result is not written into that report as a company measurement.

## Ceiling, not closed

Failing number: phone ceiling against the FARO height peak is 5.4 cm and 4.5 cm. The two walks differ by 1.1 cm. Both the 1.5 cm bar and the 1 cm spread are missed on that reference. No code change was applied to pull a ceiling onto the peak. Reproduction: `benchmarks/external/arkitscenes_42444949/comparison.json`.
