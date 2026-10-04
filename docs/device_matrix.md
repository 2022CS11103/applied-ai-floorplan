# Device matrix

The assignment asks for three tiers on iPhone 15 and newer. This table is what the code will do, and what it will not pretend to do.

| Device | Photo | Video | LiDAR | Honest accuracy |
| --- | --- | --- | --- | --- |
| iPhone 15, 15 Plus, 16, 16 Plus | yes | yes | no sensor | Photo: wall length interval at least ±8%. Video with no poses: same prior, often wider. Video with VIO poses: interval at least ±3%, wider when walls are blank. |
| iPhone 15 Pro, 15 Pro Max, 16 Pro, 16 Pro Max | yes | yes | yes | LiDAR: wall position from the scanned face, interval floor 1.2 cm. Openings only when both jambs are in the cloud. Ceiling only when a plane is actually there. |
| This Windows machine | runs all three tiers on folders | runs | runs on the supplied bundles | Cannot capture. No second device was available. No laser ground truth was taken, so the centimetre gates are not claimed as passed. |

## What each tier is allowed to read

| Tier | Reads | Does not read |
| --- | --- | --- |
| lidar | depth, confidence, poses, IMU, RGB | nothing from a server |
| video | RGB, and poses if the file has them | depth images, even if they sit in the same folder |
| photo | the stills and `adjacency.json` | depth, poses, the LiDAR plan |

Photo scale is the chest-height prior in the protocol (1.40 m), because stills have no metric sensor. That is why the photo interval is not allowed to collapse under ±8%.

## Surfaces the brief calls out

| Surface | What the pipeline does |
| --- | --- |
| Mirror, glass | Confidence 0 and 1 returns are dropped on the LiDAR tier. A glass wall often becomes a gap. A gap without wall on both sides is not called a door. |
| Wet-look floor | The floor plane is a RANSAC fit. Specular holes in the floor do not move the plane if the rest of the floor agrees. |
| Low light | RGB damage detection abstains when the local contrast is only a lighting gradient. Geometry does not need the RGB. |

## Sample data this submission actually ran

| Capture | What it is | Tier run |
| --- | --- | --- |
| `single_room/c00a170fe1` | supplied bundle, no ceiling in view | lidar |
| `single_scan_with_ceiling/c7d28f72c6` | supplied bundle, ceiling in view | lidar |
| synthetic box in `python run.py self-check` | 4.00 x 3.00 x 2.50 m, door 0.80 m | lidar logic, no phone |

`single_scan_floor_only` is the same kind of bundle. It is on disk. It is not a separate architectural claim beyond the two runs above.
