# Compliance matrix

| Requirement | File | Artifact | Status |
| --- | --- | --- | --- |
| Capture route, one page, non-engineer | `docs/capture_protocol.md` | protocol | done, route 2 |
| Device matrix | `docs/device_matrix.md` | table | done |
| One command per capture | `run.py` | `python run.py run --capture ... --tier ... --out ...` | done |
| LiDAR tier | `cozmo_scan/fuse.py`, `cozmo_scan/layout.py` | `runs/single_room_lidar`, `runs/ceiling_lidar` | ran |
| Video tier, no depth | `cozmo_scan/reconstruct.py` | `runs/single_room_video` | ran, refused a plan (611 points) |
| Photo tier, 2–8 stills, stitch by sketch | `cozmo_scan/reconstruct.py`, `cozmo_scan/pipeline.py` | code path | not closed on this sample |
| JSON schema | `schema/floorplan.schema.json` | `plan.json` | done |
| Dimensioned plan | `cozmo_scan/render.py` | `plan.png`, `plan.svg` | done |
| Confidence interval on every length | `cozmo_scan/geometry.py` `meas` | fields `ci95_low`, `ci95_high` | done |
| Ceiling, area, walls, openings | `plan.json` rooms | see benchmark | ceiling only when a plane exists |
| Damage class and extent | `cozmo_scan/damage.py` | empty on these scans | abstained, not filled in |
| Concealed-damage rule that fired | `concealed_damage` in JSON | rules present, `fired: false` | evaluated |
| Scope lines keyed to surfaces | `scope_line_items` | empty when there is no damage | correct |
| Multi-room adjacency | `property.adjacencies` | shared walls on `c00a170fe1` | partial |
| Drift method and ablation | `drift` in JSON | anchor on vs off | done |
| Repeatability | `runs/repeat_before`, `runs/repeat_after` | `repeatability.json` | proxy, see fix loop |
| Fix loop, before and after regenerable | `docs/fix_loop.md` | both commands | shipped, gate moved |
| Head to head vs a scanning app | — | — | blocked, no phone |
| Laser ground truth | — | — | blocked, no device |
| Technical report ≤ 6 pages | `docs/technical_report.md` | report | done |
| Raw sample data | local folders, gitignored | depth, confidence, video, poses | on disk, not in git |
| Process history | git log | commits | this repo |
