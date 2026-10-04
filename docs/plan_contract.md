# Plan contract

LiDAR, video, and photo write the same `plan.json`. The file is checked against `schema/floorplan.schema.json`. Reconstruction is different per tier. The room graph below is not.

`python run.py` applies this shape after the tier has finished. It does not add a room, a wall, or an adjacency that the tier did not already produce. A missing ceiling stays `null`. A length that the tier measured keeps the sigma that tier computed.

## Document

| Field | Meaning |
| --- | --- |
| `status` | `ok` or `degraded`. An empty room list is always `degraded`. |
| `degraded_reasons` | Reasons that tier actually checked. Present on every tier, including a successful LiDAR plan, where it is `[]`. |
| `property.rooms` | Rooms sorted by `room_id`. |
| `property.adjacencies` | Links sorted by `room_a`, then `room_b`. |

## Room

| Field | Meaning |
| --- | --- |
| `id`, `room_id` | The same deterministic id. A photo property uses the folder name. A single layout uses `room_0`, `room_1`, … in grid order. |
| `polygon_m`, `floor_polygon` | The same floor polygon, in metres, XZ. |
| `floor_area_m2` | Measurement object. |
| `ceiling_height_m` | Measurement object, or `null` when the ceiling was not observed. |
| `walls` | Each wall has `length_m` as a measurement. `height_m` is the ceiling measurement or `null`. |
| `openings` | `width_m` is a measurement. Sill and head are `null` when they were not seen. |
| `measurements` | The same objects as the fields above, gathered in one place. |
| `status`, `reasons` | The room itself. A room in the list has a closed polygon. A property can still be `degraded` because the rooms do not connect. |

A measurement object is always:

```json
{"value": 0.0, "sigma": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
```

The numbers come from `geometry.meas` and the tier floor (LiDAR 1.2 cm, video 4 cm and 3%, photo 8 cm and 8%). This contract does not invent a sigma for a quantity the tier left unobserved.

## Adjacency

A link exists only when two floor polygons share an edge.

| Field | Meaning |
| --- | --- |
| `a`, `room_a` | One room id. |
| `b`, `room_b` | The other room id. Both ids are in `property.rooms`. |
| `kind`, `relationship` | `shared_wall` for a geometric link. |
| `source` | `geometry`. A handwritten `adjacency.json` is not a source. |
| `gap_m` | Distance between the facing edges, metres. |
| `shared_length_m` | Along-wall overlap of those edges, metres. This is geometry of a link that already exists. It is not a confidence interval. |

If the polygons do not share an edge, the adjacency list stays empty. The plan does not draw a connector to force one property.

## Damage

Damage hangs off the room that already exists. A region carries `room_id` and `surface_id`. It does not contain a second copy of the floor polygon. The extent is `extent_m2`, the same measurement object as `extent`. `confidence` is null: this detector has a color residual, not a calibrated probability. Classes, the concealed-damage rules, and the synthetic fixture are in `docs/damage.md`.

## What is not claimed

Matching this schema is not a laser or tape check. Footprint, wall length, and opening gates stay blocked until a physical measurement exists.
