# Technical report

## Architecture

A capture becomes a point cloud, the cloud is stood upright on its floor, and the plan is read off the walls in that plane. LiDAR, video, and photos differ only in how the cloud is built and in how wide the interval is allowed to be. One JSON schema comes out of all three.

LiDAR back-projects the 256×192 depth image with the per-frame intrinsics, scaled from the 1920×1440 RGB matrix, and transforms it by the pose. Confidence below 2 is dropped. That is the glass and mirror cut: those pixels are the low-confidence ones. Video triangulates ORB matches with the same poses and never opens a depth file. Photos run a short sequential reconstruction and scale it so the cameras sit 1.40 m above the floor, which is the chest height in the protocol. If that plane is not there, the photo run is marked degraded instead of being given a scale.

## Frame

The quaternion is camera-to-world. The camera that fits this logger is OpenCV: X right, Y down, Z forward. After that transform, Y is up, the camera path is almost level, and the camera sits about 1.4 m above a dense horizontal plane. The ARKit camera axis (looking down −Z) was tried. It makes the vertical span about 5 m. That was rejected because it is not a room. The choice is checked by `python run.py self-check` and by the IMU: specific force on both samples is 1.00 g, so the accelerometer is in g and the phone was not in free fall.

## Floor, drift, ceiling

A RANSAC plane is fit to the low points and rotated onto +Y. The walk is then split into four time chunks and each chunk's floor is shifted to zero. A slow vertical drift otherwise smears the wall band and the ceiling. The ablation is the same cloud with that rotation and those shifts turned off. On `c00a170fe1` the footprint goes from 10.96 m² to 14.00 m². On the ceiling scan it goes from 6.52 m² to 9.27 m².

A loop was also estimated: wall points from the end of the walk against wall points from the start, yaw and translation only. It is applied only when the correction is small and the error actually drops. On both samples the correction was too big to be a loop, so it was not applied, and the JSON says why. That is the drift row. Poses are not taken as-is, and a failed loop is not hidden.

Ceiling height is the mode of returns above 1.85 m inside the room, and only when that mode is a tight plane (standard deviation under 4 cm). `c00a170fe1` never gets there. The ceiling scan does: 2.27 m and 2.47 m in the two rooms that closed.

## Walls and rooms

A column of points is a wall when it spans at least 0.65 m and has returns between 0.45 m and 1.60 m. Floor plus ceiling in one cell is not a wall. A table is not a wall. The remaining points are rotated to a Manhattan frame. The angle comes from local PCA on the occupancy grid, folded into a right angle. A silhouette gradient was tried first. It locked onto the bounding box, which is axis-aligned even when the room is rotated about 24° in this capture, and the plan came out empty. Local PCA gives 23.5° on the first sample and 28.5° on the second.

Wall positions are peaks of that histogram. Peaks closer than 0.32 m are one surface seen twice. Weak peaks are dropped so a sofa does not become a room. A grid cell is a room when it is at least 1.15 m on a side, its core is not solid with structure, and at least three of its four edges were actually observed along that edge. The last test is what stopped the ceiling scan drawing a lattice of rooms in the gaps between lines. It also drops real space that was only partly scanned. On the ceiling walk, two rooms closed and the connection between them did not. The plan shows two rooms and an empty adjacency list.

The length of a wall is the distance between the cross-walls. The earlier version used the length of the points that were seen. Even and odd frames of one walk then disagreed by 5 cm on a wall they both saw. The cross-wall distance disagrees by 0.5 cm. That is the fix loop.

## Intervals

A 95% half-width is 1.96 times the scatter of the points that define the wall, and never tighter than a floor that depends on the tier: 1.2 cm for LiDAR, 3% and at least 4 cm for video, 8% and at least 8 cm for photos. The floors exist so a thin tier cannot publish a millimetre. Area uncertainty grows with the perimeter times the wall-position scatter.

## Openings, damage, scope

An opening is a gap in the walking band, 0.68 m to 1.20 m, with wall on both sides. A missing end of a scan is not a door. Head height is left empty when the header was not seen. On the synthetic box the door is recovered at 0.90 m against 0.80 m. The 5 cm bin is the reason, and it is not good enough for the 2 cm opening gate.

Damage is a local color residual on the wall, in 10 cm cells, against the neighbouring cells so a lighting gradient does not fire. A region has to be clearly darker, at least 0.12 m², and compact. Moisture is that, plus a yellow shift, at the base of the wall. Two concealed rules are always written down: stain or moisture in the bottom 30 cm, and stain under a window. On these samples nothing fired, and the scope list is empty. There was nothing staged to find.

## Error budget

| Term | Size on these files | Where it goes |
| --- | --- | --- |
| Depth noise and the 2.5 cm voxel | about 1 cm | wall-position sigma |
| Floor tilt removed | 0.49° and 0.03° | ablation, not left in the plan |
| Vertical chunk drift | recorded per chunk in the JSON | taken out |
| Unclosed loop | not applied | horizontal drift remains |
| Opening bin | 5 cm | door width, the known miss |
| Photo scale prior | 1.40 m assumed | ±8% floor, and only if a floor plane exists |
| Video sparsity | 611 points here | no plan, rather than a bad plan |

## Failure modes

Mirrors and glass are dropped with confidence, and a glass wall can look like a missing wall. The plan would rather omit it than invent a door. A room scanned from the middle, with one side never seen, does not close; the ceiling walk shows that. Furniture parallel to a wall can still add a ridge. The weak-peak cut reduces it and does not eliminate it: even frames and odd frames still disagree on the room count. Low light hurts color and does not hurt the LiDAR lengths. Wet floors punch holes in the floor cloud; the plane fit tolerates that until the inliers collapse.

## What this submission is not

It is not a laser-checked benchmark, and it is not a win over Polycam. Both of those need a phone this project did not have. The walk-in, if it happens, uses the protocol page on a phone the interviewer brings. The command is the same one that produced `runs/`.
