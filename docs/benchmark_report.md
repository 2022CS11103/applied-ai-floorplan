# Benchmark

No laser, no tape, no second phone. The gates that need a tape are marked blocked. They are not marked passed. The numbers below are what this machine could regenerate from the supplied bundles and from the synthetic box.

## Synthetic box (`python run.py self-check`)

| Quantity | Truth | Pipeline |
| --- | --- | --- |
| Floor | 4.00 m by 3.00 m | 3.999 m and 2.998 m |
| Area | 12.00 m² | 11.991 m² |
| Ceiling | 2.50 m | 2.498 m |
| Door | 0.80 m | 0.863 m |

The door is 6.3 cm wide of the truth. Width is measured from column centers, not from a 5 cm bin edge. The sheet still steps by about 6 cm, so the ≤2 cm opening gate is not met. Real scans below report an opening only when both jambs are present, and none were, so there is no real opening to score.

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

## Head to head

Not run. Polycam, Magicplan, and the others need the same rooms on a phone, and this machine cannot capture. A made-up comparison table would be the benchmark-avoidance failure the scoring table names. The protocol in `docs/capture_protocol.md` is what a phone with the app would follow.

## Gates

| Gate | Result |
| --- | --- |
| Opening width ≤ 2 cm on ≥ 85% | Blocked. No tape. Synthetic door is 6.3 cm off (0.863 m vs 0.80 m). The opening benchmark is 34/62 ≤ 2 cm, which is under 85%. Real scans: no opening emitted. A miss or a phantom would count as a miss. |
| Ceiling ≤ 1.5 cm, repeat spread ≤ 1 cm | Blocked. No tape and no second capture. `c00a170fe1` and the floor-only scan leave ceiling null. `c7d28f72c6` reports 2.271 m and 2.466 m from observed planes. |
| Repeatability 1 cm or 0.5% | Blocked as an assessment gate. Even/odd frames of one walk are a proxy, not two captures. |
| Drift, with an ablation | Floor anchor on vs off, numbers above. Loop checked and not applied. Poses are not used as-is on LiDAR. |
| Photo stitch of a whole property | Synthetic fixture stitches with adjacency and no overlap. Physical ±8% footprint is blocked: no calibrated photo capture. |
| Photo ±8%, video ±3%, with intervals | Interval floors are in the code. Video did not close, so it did not publish a wall length. Photo fixture metres are not a tape. |
