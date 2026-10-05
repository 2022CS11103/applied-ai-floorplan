# Pipeline results

Status means geometry, not "the process exited 0".

`PASS` is a metric inside its written threshold.
`FAIL` means the command ran and the geometry or metric missed.
`PARTIAL` means a real product exists and a required piece is missing.
`BLOCKED` means the files needed for that gate are not available.
`NOT APPLICABLE` means the dataset is the wrong kind of input for that tier.

No public set below was turned into fake LiDAR.

## Company captures (real data, no tape)

| Dataset | Scene | Tier | Input | GT | Command | Status | Rooms | Wall error | Opening | Ceiling | Area | Adjacency | Drift | Requirement validated | Not validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Company sample | c00a170fe1 | lidar | depth, confidence, odometry, K, IMU | none | `python run.py run --capture single_room\c00a170fe1 --tier lidar --out runs\sample_single_room` | PARTIAL | 3 predicted, GT unavailable | unavailable | 0 openings, no GT | not observed | 7.097, 4.053, 2.850 m², no survey | 2 shared walls produced, not surveyed | anchor-on footprint 14.000 m², anchor-off 10.965 m² | one command emits a multi-room plan with CI | centimetre gates |
| Company sample | c7d28f72c6 | lidar | same, ceiling in the cloud | none | `--capture single_scan_with_ceiling\c7d28f72c6 --tier lidar` | PARTIAL | 2 / unavailable | unavailable | none | 2.271 m and 2.466 m, no tape | 3.181 and 6.088 m² | none | anchor-on 9.269 m², anchor-off 13.532 m² | ceiling is emitted when the scan sees it | ceiling ≤1.5 cm |
| Company sample | 1a8384c3f6 | lidar | floor-biased cloud | none | `--capture single_scan_floor_only\1a8384c3f6 --tier lidar` | PARTIAL | 1 / unavailable | unavailable | none | not observed | 3.381 m² | none | one closed cell | a floor-only scan can close the supported cell | extra rooms were not invented; tape still missing |
| Company sample | c00a170fe1 | video | rgb.mp4 + odometry | none | `--tier video --out runs\sample_single_room_video` | FAIL | 0 | n/a | n/a | n/a | n/a | n/a | n/a | reconstruction runs: 26,381 raw points, 9,688 after voxel, 72 frames | room closure. Manhattan theta 40.5°, 1 vertical line, 2 horizontal. `no_room_closure` |
| Company sample | c7d28f72c6 | video | rgb.mp4 + odometry | none | `--tier video` | FAIL | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 1,519 raw / 1,138 voxel points | 0 wall lines. Frames are close-ups, not a room walk |
| Company sample | synthetic fixture | photo | 15 stills in 4 folders | synthetic rectangles | `--capture fixtures\synthetic_photo_property --tier photo` | PARTIAL | 4 / 4 | synthetic only | none in the fixture | none | footprint 33.331 vs 33.84 m² outer rectangles | 3 geometry edges, overlap 0 | scale is the 1.40 m chest-height prior, labeled estimated | stitch code path | physical ±8% and tape |

Root cause of the video failure: sequential frame decode matches the pose timestamps, and the poses stay metric. ORB triangulation on these close, low-texture frames sprays streaks. Vertical columns do not form two directions, so the layout correctly refuses a room. That is not a successful plan.

## External sets that were executed

| Dataset | Scene | Tier | Input | GT | Command | Status | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARKitScenes | 42444949, visit 421337 | lidar | Apple lowres depth, confidence, pincam, traj. No IMU, no RGB video. | FARO `highres_depth` (laser depth in the camera). Not a vector plan. | `python run.py run --capture datasets\_inspect\arkitscenes\capture\42444949 --tier lidar` | PARTIAL | 26.4 s. status ok, 1 room, 4.025 m by 3.710 m, area 14.933 m², ceiling 3.036 m. Same extractor on the FARO cloud: 4.106 m by 3.642 m, area 14.953 m², ceiling peak 3.090 m. Wall error 6.8–8.1 cm (about 2%). Area error 0.020 m². Ceiling error 5.4 cm, outside 1.5 cm. Apple-vs-FARO depth median 2.4 cm. Openings unavailable. Assessment gate blocked. |
| ARKitScenes | 42444946, same visit | lidar | same | same | `--capture ...\42444946 --tier lidar` | PARTIAL | 20.4 s. status ok, 1 room, 4.122 m by 3.695 m, area 15.230 m², ceiling 3.025 m. FARO height peak 3.070 m, error 4.5 cm. FARO cloud did not close a room: Manhattan theta 67.5°, 1 vertical line and 2 horizontal, so the extractor refused a room. The phone cloud at 68.5° had 2 and 2 and closed. Wall length versus that FARO cloud is unmatched. Depth median 2.1 cm. |
| ARKitScenes | visit 421337, both videos | lidar repeat | two walks | each other, not tape | both commands above | FAIL | Long walls differ by 9.7 cm (2.4%). Short walls differ by 1.5 cm (0.41%). The 1 cm or 0.5% bar fails on the long wall. Ceiling spread 1.1 cm. |
| HouseLayout3D | 2t7WUuJeko7 | not a pipeline tier | doors, windows, layout PLY, plane equations | external CAD, derived | `python -m datasets.houselayout3d_adapter` | PARTIAL | 6 floor polygons kept, smaller horizontals listed and not called rooms, 4 door links. `assessment_gate` blocked. No `run.py` call, because RGB and depth are not in the release. |

The FARO wall lengths are `DERIVED_FROM_DATASET`: the same `build_layout` ran on points backprojected from `highres_depth`. That checks whether the phone cloud and the laser depth describe one rectangle. It does not certify a tape.

## Not executed

| Dataset | Why |
| --- | --- |
| ARKitScenes, rest of the release | Only videos 42444949 and 42444946 were downloaded. The 3DOD pack is 623.4 GB. These two do not contain door widths. |
| ScanNet | `.sens` files require an approved request. |
| ScanNet++ | iPhone and Faro files require an approved request. |
| ZInD | Academic approval, about 40 GB of panoramas. Cannot be passed to the LiDAR tier. |
| LaserDoors | FTP only after an email form. Single doors, so it cannot score a floor plan. |
| ISPRS TUB1/TUB2 | Public, but the input is a laser point cloud and a trajectory, not depth frames. Feeding that cloud to `fuse_lidar` would be a fake phone run. |
| StructScan3D | GitHub page checked; the frame archive was not in the repo. |
| Structured3D, Matterport3D, 3D-FRONT, Hypersim, Replica, S3DIS, Redwood | Too large, access-gated, or synthetic, and none is a phone log plus tape. |

## Opening numbers already measured

Synthetic self-check door: predicted 0.863 m, fixture 0.800 m, absolute error 0.063 m. Gate is 0.020 m. One fused grid of the synthetic walls is 39 of 62 openings within 2 cm (62.9%). The older 34/62 figure is the count before the sparse path. Company scans contain no surveyed openings, so real opening error is unavailable.

The 6 cm residual matches the station spacing of the wall columns (about 5–6 cm) plus the 2.5 cm fusion voxel. Tightening the histogram bin to force 0.800 m would pass the fixture and lie on a real jamb that is not exactly on a column. No threshold was changed in this audit.
