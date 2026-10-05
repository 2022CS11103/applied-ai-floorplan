# Validation summary

The assessment is not fully benchmarked. The gates that need a tape, a second physical walk, phone stills, or a Polycam/Magicplan export are still blocked.

## Best datasets

1. ARKitScenes videos `42444949` and `42444946` (visit `421337`): real Apple LiDAR plus FARO depth, and two walks of one venue.
2. The company captures, for the LiDAR command on the bundle the assessment shipped, including IMU.
3. HouseLayout3D annotations, for an external wall and opening drawing. One scene was converted and labeled derived.
4. ZInD and LaserDoors, after their approval steps, for layout boxes and single-door widths. Neither is a LiDAR walk.

## Coverage

| Need | Public set that has it | Ran here |
| --- | --- | --- |
| Phone depth + poses | ARKitScenes, ScanNet++, company sample | company sample and two ARKitScenes videos |
| Video frames + poses | company sample, Redwood, ScanNet | company sample only |
| 2–8 pinhole stills per room | not found | synthetic fixture only |
| Vector openings | HouseLayout3D, ZInD, LaserDoors, Structured3D | HouseLayout3D annotations only |
| Laser depth on the same walk as the phone | ARKitScenes `highres_depth` | two videos. Depth, not a tape of each wall |
| Repeated walk of the same room | ARKitScenes visit 421337 | both videos ran |
| Incumbent export | none | no |

## What works

- LiDAR on `c00a170fe1`: status ok, 3 rooms.
- LiDAR on `c7d28f72c6`: status ok, 2 rooms, ceilings 2.271 m and 2.466 m.
- LiDAR on `1a8384c3f6`: status ok, 1 room, 3.381 m². The earlier empty plan was a wall filter dropping the long supported lines.
- Photo stitch on the synthetic 4-room fixture: status ok, overlap 0. Scale is the 1.40 m prior.
- HouseLayout3D adapter writes a blocked external case and lists the polygons it refused to call rooms.
- ARKitScenes LiDAR on both videos: status ok, one room each, about 4.0 m by 3.7 m, ceiling about 3.03 m. Floor tilt after the Z-up to Y-up axis swap is 0.33° and 0.16°. On `42444949` the phone area and the FARO-layout area differ by 0.020 m².

## What fails

- Video on both company captures: `no_room_closure`.
- Opening width on the synthetic door: 6.3 cm, and 39 of 62 synthetic openings (62.9%) within 2 cm on one fused grid. ARKitScenes does not publish door widths, and neither plan emitted an opening.
- Ceiling against the FARO height peak: 5.4 cm on `42444949` and 4.5 cm on `42444946`. The 1.5 cm bar is missed. The peak is a histogram of laser depth, not a tape, so the assessment gate stays blocked.
- Repeatability of the two walks: the long wall moves 9.7 cm (2.4%). That misses 1 cm and 0.5%. The short wall moves 1.5 cm (0.41%), which is inside 0.5% and outside 1 cm.
- On `42444946` the FARO cloud did not close a room, so there is no laser-side wall length for that walk.
- Drift as an accuracy claim on the company sample: the on/off footprints differ, which shows the anchor moves the plan, and does not show which one is true. On `42444949` the anchor barely moved the footprint (14.933 m² vs 14.930 m²).

## Fixes in this tree

- Layout keeps a long, well-supported wall instead of dropping it for a short clutter ridge. Floor-only went from 0 rooms to 1 room. The other two LiDAR scenes kept their room areas.
- Photo plans now emit concealed flags and scope lines. On the undamaged fixture every flag is unfired and the scope list is empty.
- HouseLayout3D conversion records `field_status` and does not label a small polygon as a door.

No opening, fusion, or video threshold was edited to manufacture a pass.

## Gates

| Gate | State |
| --- | --- |
| One command, schema, CI, multi-room LiDAR plan | validated on the company sample |
| Opening ≤2 cm on ≥85% | blocked, and the synthetic number fails |
| Ceiling ≤1.5 cm and repeat spread ≤1 cm | blocked as a tape gate. Measured FARO-peak errors are 5.4 cm and 4.5 cm. The two walks differ by 1.1 cm |
| Repeatability ≤1 cm or 0.5% | not passed. Two ARKit walks of visit 421337 differ by 9.7 cm on the long wall. Not a tape protocol |
| Video walls ≤3% | failed to produce a room on the only real video |
| Photo ±8% and stitch | synthetic only; physical stills blocked |
| Incumbent ≥70% | blocked |
| Damage on a real furnished room | blocked; synthetic fixture only |

## Still required before submission can claim the PDF gates

- A tape, or a vector survey, of openings. FARO depth does not label a door width.
- Phone stills, 2–8 per room, for at least three rooms.
- One Polycam or Magicplan export of the same capture.
- The ceiling and repeatability numbers above already miss their bars on this visit. A tape would not be expected to turn a 5 cm ceiling gap into a pass without a code change, and no threshold was loosened to hide that.
