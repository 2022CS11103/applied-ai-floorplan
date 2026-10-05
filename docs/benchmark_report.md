# Benchmark

No laser, no tape, no second phone. The gates that need a tape are marked blocked. They are not marked passed. The numbers below are what this machine could regenerate from the supplied bundles and from the synthetic box.

## Synthetic box (`python run.py self-check`)

| Quantity | Truth | Pipeline |
| --- | --- | --- |
| Floor | 4.00 m by 3.00 m | 3.999 m and 2.998 m |
| Area | 12.00 m² | 11.991 m² |
| Ceiling | 2.50 m | 2.498 m |
| Door | 0.80 m | 0.863 m |

The door is 6.3 cm wide of the truth. That sheet is an empty hole: no second surface was scanned at the jamb, so the width is the center of the unobserved sample interval, and a 6 cm column can miss by about that much. Where a jamb face is scanned, the width is the along-track position of that depth step. The ≤2 cm opening gate is not met on one frame of the 62-case along-track set (39/62). The five one-hit walls are included in that 39. The 19 unobserved intervals are not. Real scans below report an opening only when both jambs are present, and none were, so there is no real opening to score.

The same 62 openings, each camera shifted by a fraction of the column pitch. Edges are estimated per frame in wall-local coordinates and aggregated as the overlap of the unobserved brackets. They are not voxel-fused first. One-hit and two-hit bins stay off the 3-hit wall test. A separate sparse path accepts them when neighboring bins continue the wall, and labels that jamb `sparse_wall_termination` at confidence 0.52.

| Frames | Within 2 cm | MAE | Median | p95 | Worst | Undetected | Unobserved intervals | Deleted columns | Sparse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 39/62 (62.9%) | 1.55 cm | 1.31 cm | 4.42 cm | 10.05 cm | 0 | 0/19 | 0/4 | 5/5 |
| 2 | 54/62 (87.1%) | 0.74 cm | 0.48 cm | 2.29 cm | 4.13 cm | 0 | 12/19 | 3/4 | 5/5 |
| 3 | 61/62 (98.4%) | 0.48 cm | 0.24 cm | 1.70 cm | 2.29 cm | 0 | 18/19 | 4/4 | 5/5 |
| 5 | 62/62 (100%) | 0.38 cm | 0.27 cm | 1.30 cm | 1.86 cm | 0 | 19/19 | 4/4 | 5/5 |

Before the sparse path and the bracket overlap, one frame was 34/62 and three frames were 56/62. The five one-hit cases are the whole of that single-frame change. Three frames now leave one miss: `far_6cm_0.70` at 2.29 cm. Its third view has a 5 cm bin gap of 0.65 m and a midpoint of 0.66 m, so that view is not called a door. Five frames include a view whose bin gap is still 0.70 m and whose midpoint is 0.664 m. The overlap of those brackets is inside 2 cm, and it is not snapped to 0.70 m.

This table is the synthetic capture schedule. It is not a tape measurement, and it is not what a real walk produces. `fuse_lidar` still voxel-downsamples every frame into one cloud before `build_layout`. The sparse path does run on that fused cloud. The per-frame brackets do not, because the sub-voxel returns were already discarded. A single fused grid remains 39/62.

## LiDAR, `c00a170fe1` (no ceiling in the cloud)

20.2 s. 215 of 1715 depth frames. IMU specific force 1.003 g.

| Room | Area | Sides | Ceiling |
| --- | --- | --- | --- |
| room 0 | 7.10 m² [6.71, 7.48] | 2.40 m by 2.95 m | not observed |
| room 1 | 4.05 m² [3.72, 4.39] | 2.40 m by 1.69 m | not observed |
| room 2 | 2.85 m² [2.55, 3.15] | 2.40 m by 1.19 m | not observed |

The three rooms share the 2.40 m direction. room 0 shares a wall with room 1 (gap 0). room 1 shares a wall with room 2 (gap 9.5 cm, the two faces of that wall). Highest returns sit near 1.7 m, so a ceiling height would have been a guess. It is omitted.

Damage: 0 regions. The detector requires a local darkening, not a lighting gradient. Concealed-damage rules were evaluated on every wall and none fired. The sample has no staged damage. Inventing stains to fill the contract would have been the failure mode the brief describes.

Openings: 0. A gap is a door only between 0.68 m and 1.20 m with at least 0.35 m of wall on both sides. These scans do not present that pattern. A phantom door counts as a miss, so the threshold stays high.

## LiDAR, `c7d28f72c6` (ceiling in view)

22.8 s. 153 of 459 depth frames.

| Room | Area | Sides | Ceiling |
| --- | --- | --- | --- |
| room 0 | 3.18 m² [2.85, 3.51] | 1.79 m by 1.78 m | 2.27 m |
| room 1 | 6.09 m² [5.62, 6.56] | 3.42 m by 1.78 m | 2.47 m |

The walk is several metres longer than these two rooms. Cells that did not have support on three sides were not drawn. The two rooms that did close do not share a detected wall, so the adjacency list is empty. That is a missed stitch, not a claim that the property is two islands. Ceiling 2.47 m is the sharp plane in the taller room. 2.27 m is the plane in the smaller one. Both are observed peaks, not a default storey height.

## Drift ablation

Floor-plane anchor on, against a single floor-height subtraction.

| Capture | Area with anchor | Area without | Floor tilt removed |
| --- | --- | --- | --- |
| c00a170fe1 | 14.00 m² | 10.96 m² | 0.49° |
| c7d28f72c6 | 9.27 m² | 13.53 m² | 0.03° |

The area moves because the wall band is defined in metres above the floor. A tilted or drifting floor pulls that band off the walls and the plan changes. Loop closure was estimated and not applied: the rigid correction was too large to be a return to the start, and the report says so. Poses were not used as-is.

## Repeatability

See `docs/fix_loop.md`. After the fix, matched walls on an even/odd split agree to 0.52 cm. This is not two physical captures.

## Video tier on `c00a170fe1`

The command does not read depth. Triangulation with the logged poses keeps 26,381 points and 9,688 after a 3 cm voxel. The cloud is streaks on close-up surfaces (tiles, appliances, floor), not a pair of wall lines, so the plan status is `degraded` with reason `no_room_closure` and zero rooms. No dimensions were invented. A video interval, when a room does close, is not allowed under ±3%.

## Photo tier

Per-room folders stitch by shared edges, not by `adjacency.json`. The labelled fixture `fixtures/synthetic_photo_property` (3 rooms plus a connector, 15 stills) writes one plan: 4 closed rooms, 3 geometry links, overlap 0 m², footprint 33.331 m². That fixture is not a tape, and the company sample is a LiDAR walk, not 2–8 stills per room. Physical photo ±8% stays blocked.

## ARKitScenes visit 421337

Two Validation videos, `42444949` and `42444946`, downloaded from the raw set. Not the 623.4 GB pack. Each has Apple LiDAR depth, confidence, intrinsics, and a trajectory, plus FARO depth projected into the camera (`highres_depth`). There is no IMU file and no door-width label. The trajectory is Z-up. The adapter swaps that to Y-up with a fixed axis change, then the existing LiDAR command runs unchanged.

| Video | Runtime | Plan | Against FARO |
| --- | --- | --- | --- |
| 42444949 | 26.4 s | 1 room, 4.025 m by 3.710 m, area 14.933 m², ceiling 3.036 m | same extractor: 4.106 m by 3.642 m, area 14.953 m². Wall error 6.8 cm and 8.1 cm. Ceiling peak 3.090 m, error 5.4 cm. Depth median 2.4 cm |
| 42444946 | 20.4 s | 1 room, 4.122 m by 3.695 m, area 15.230 m², ceiling 3.025 m | height peak 3.070 m, error 4.5 cm. The FARO cloud did not close a room (1 vertical line, 2 horizontal). Depth median 2.1 cm |

The two plans disagree by 9.7 cm on the long wall and 1.5 cm on the short wall. Ceiling spread is 1.1 cm. Openings were not scored. The assessment gate stays blocked: FARO depth is not a tape of each wall and opening. Details are in `benchmarks/external/arkitscenes_42444949/comparison.json`.

## Head to head

Not run. Polycam, Magicplan, and the others need the same rooms on a phone, and this machine cannot capture. A made-up comparison table would be the benchmark-avoidance failure the scoring table names. The protocol in `docs/capture_protocol.md` is what a phone with the app would follow.

## Gates

| Gate | Result |
| --- | --- |
| Opening width ≤ 2 cm on ≥ 85% | Blocked. No tape. Synthetic door is 6.3 cm off (0.863 m vs 0.80 m). One fused grid is 39/62 within 2 cm. Real scans: no opening emitted. A miss or a phantom counts as a miss. |
| Ceiling ≤ 1.5 cm, repeat spread ≤ 1 cm | Blocked as a tape gate. On ARKitScenes the phone ceiling is 5.4 cm and 4.5 cm from the FARO height peak, and the two walks differ by 1.1 cm. |
| Repeatability 1 cm or 0.5% | Not passed on visit 421337: the long wall moves 9.7 cm between the two Apple walks. Even/odd frames of one company walk remain a proxy. |
| Drift, with an ablation | Floor anchor on vs off, numbers above. Loop checked and not applied. Poses are not used as-is on LiDAR. |
| Photo stitch of a whole property | Synthetic fixture stitches with adjacency and no overlap. Physical ±8% footprint is blocked: no calibrated photo capture. |
| Photo ±8%, video ±3%, with intervals | Interval floors are in the code. Video did not close, so it did not publish a wall length. Photo fixture metres are not a tape. |

## Assessment inventory

`python run.py assessment-gate --manifest benchmarks/manifests/example.json` writes `assessment.json` and `assessment.md`. The example manifest has empty paths, so every gate is BLOCKED. That command does not invent a Polycam export, a tape, or a second walk.

A. Dataset inventory. Company bundles are real phone logs without tape. ARKitScenes `42444949` and `42444946` are real Apple LiDAR with FARO depth, not a tape of each opening. HouseLayout3D is an external CAD drawing. `fixtures/synthetic_photo_property` and `fixtures/synthetic_damage` are synthetic.

B. Capture protocol. `docs/capture_protocol.md`. Route 2, no TestFlight build.

C. LiDAR. Company sample closes rooms. The shipped opening path is `opening_evidence: voxel_fused`. Per-frame brackets are a separate API (`detect_openings_multiframe`) and are not what this command scores.

D. Video. Both company runs are `degraded` / `no_room_closure`. No wall length is published, so the ±3% gate is BLOCKED.

E. Photo. Synthetic property stitches. Validation rejects fewer than 2 stills, more than 8, a missing room, a repeated filename across rooms, an unsupported suffix, and an estimated scale. The 1.40 m prior stays estimated.

F. Openings. Fused synthetic grid 39/62. Five-frame synthetic schedule 62/62. Neither is a tape gate. Misses and phantoms reduce the pass rate.

G. Ceiling. FARO peaks miss 1.5 cm (5.4 cm and 4.5 cm). Spread between walks is 1.1 cm. Classification on that pair is a miss against the reference, and the PDF tape gate stays BLOCKED.

H. Repeatability. Long wall 9.7 cm. Status FAIL on those two walks. Unmatched walls are listed, not dropped.

I. Damage. Synthetic stain and moisture only. Physical gate BLOCKED.

J. Incumbent. No export. BLOCKED. The 70% bar is not filled in.

K. Drift. Anchor on/off footprints are in the LiDAR plan. Loop was checked and not applied.

L. Fix loop. `docs/fix_loop.md`. The repeatability proxy improved. It is not two physical walks.

M. Known failures. Video does not close. Opening single-frame rate is under 85%. Ceiling reference errors are 5.4 cm and 4.5 cm. Repeatability long wall is 9.7 cm. `far_6cm_0.70` is still 2.29 cm at three synthetic frames.

N. Blocked gates. Tape openings, tape ceiling, physical photo scale, closed video with tape, furnished-room damage, incumbent export, and one property captured at all three tiers. FARO is not accepted as tape.

