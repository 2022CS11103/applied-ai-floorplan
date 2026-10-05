# Validation summary

The assessment is not fully benchmarked. The gates that need a tape, a second physical walk, phone stills, or a Polycam/Magicplan export are still blocked.

## Best datasets

1. The company captures, for proving the LiDAR command on real depth, poses, intrinsics, and IMU.
2. HouseLayout3D annotations, for an external wall and opening drawing. One scene was converted and labeled derived.
3. ARKitScenes and ScanNet++, if access and disk allow later. They match a phone walk. They still do not replace tape.
4. ZInD and LaserDoors, after their approval steps, for layout boxes and single-door widths. Neither is a LiDAR walk.

## Coverage

| Need | Public set that has it | Ran here |
| --- | --- | --- |
| Phone depth + poses | ARKitScenes, ScanNet++, company sample | company sample only |
| Video frames + poses | company sample, Redwood, ScanNet | company sample only |
| 2–8 pinhole stills per room | not found | synthetic fixture only |
| Vector openings | HouseLayout3D, ZInD, LaserDoors, Structured3D | HouseLayout3D annotations only |
| Laser or tape on the same walk as the phone | not in any set downloaded | no |
| Repeated walk of the same room | not in the sets above | even/odd proxy only |
| Incumbent export | none | no |

## What works

- LiDAR on `c00a170fe1`: status ok, 3 rooms.
- LiDAR on `c7d28f72c6`: status ok, 2 rooms, ceilings 2.271 m and 2.466 m.
- LiDAR on `1a8384c3f6`: status ok, 1 room, 3.381 m². The earlier empty plan was a wall filter dropping the long supported lines.
- Photo stitch on the synthetic 4-room fixture: status ok, overlap 0. Scale is the 1.40 m prior.
- HouseLayout3D adapter writes a blocked external case and lists the polygons it refused to call rooms.

## What fails

- Video on both company captures: `no_room_closure`.
- Opening width on the synthetic door: 6.3 cm, and 54.8% of synthetic openings within 2 cm.
- Every centimetre gate, because no tape exists.
- Drift as an accuracy claim: the on/off footprints differ, which shows the anchor moves the plan, and does not show which one is true.

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
| Ceiling ≤1.5 cm and repeat spread ≤1 cm | blocked |
| Repeatability ≤1 cm or 0.5% | blocked |
| Video walls ≤3% | failed to produce a room on the only real video |
| Photo ±8% and stitch | synthetic only; physical stills blocked |
| Incumbent ≥70% | blocked |
| Damage on a real furnished room | blocked; synthetic fixture only |

## Still required before submission can claim the PDF gates

- A tape or laser on the same walls, openings, and ceilings.
- A second walk of one room.
- Phone stills, 2–8 per room, for at least three rooms.
- One Polycam or Magicplan export of that same capture.
