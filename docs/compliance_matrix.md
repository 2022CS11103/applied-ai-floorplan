# Compliance matrix

Physical gates stay blocked. There is no laser or tape survey, no second walk, and no phone capture of 2–8 stills. A synthetic check is not marked as a passed gate.

| Requirement | Where | Status |
| --- | --- | --- |
| One command per capture | `python run.py run --capture ... --tier lidar\|video\|photo --out ...` | done |
| JSON to `schema/floorplan.schema.json` | `plan.json` after `apply_contract` | done |
| Rendered plan | `plan.png` next to `plan.json` | done |
| Walls, floor area, confidence interval | measurement object `value`, `sigma`, `ci95_low`, `ci95_high` | done on every closed room |
| Ceiling height | measurement when a plane is observed, otherwise null | observed on `c7d28f72c6` (2.271 m, 2.466 m). Null on `c00a170fe1` and the floor-only scan |
| Openings | width measurement, miss and phantom both fail the gate | none on the company scans. Synthetic door 0.863 m vs 0.80 m. Gate blocked |
| Stitched multi-room plan and adjacency | `property.adjacencies`, source `geometry` | LiDAR `c00a170fe1`: 3 rooms, 2 shared walls. Photo fixture: 4 rooms (3 plus connector), 3 links, overlap 0 |
| Damage class and extent in m² | `room.damage` | empty on company scans and on the photo fixture. Synthetic staged room in `fixtures/synthetic_damage` reports stain and moisture. Not a tape |
| Concealed-damage flag and the rule | `concealed_damage` on each wall | evaluated. `fired` is false when the predicate is not met |
| Scope lines keyed to surfaces | `scope_line_items` | present only for a detected region. Empty when there is no damage |
| Drift ablation, poses not used as-is | `drift.footprint_area_m2_anchor_on/off` | LiDAR `c00a170fe1`: 14.000 m² vs 10.965 m². Loop checked, not applied |
| Photo whole-property stitch | `fixtures/synthetic_photo_property` | stitches. Physical ±8% blocked |
| Video ±3% wall lengths | `runs/sample_single_room_video` | `degraded`, `no_room_closure`. 9,688 points, no invented room |
| Repeatability, two captures | `python run.py repeat` | even/odd proxy of one walk only. Assessment gate blocked |
| Benchmark set with laser/tape on the same rooms at three tiers | — | not built. Company sample has no tape. Evaluator in `cozmo_scan/benchmark_eval.py` stays `assessment_gate: blocked` without `provenance: laser_tape` |
| Head to head vs Polycam or Magicplan | — | not run. No second app capture |
