# Compliance matrix

Source: `Applied_AI_Case_Study.pdf` (Aug 2026, 6 pages).

Implementation status below uses DONE, PARTIAL, BLOCKED, and NOT IMPLEMENTED. DONE means the code and the command exist and the tests exercise them. It does not mean a physical gate passed. The measurement result of a gate is still only PASS, FAIL, BLOCKED, or DEGRADED. A missing tape, a FARO file, or the 1.40 m photo prior stays BLOCKED. FAIL means a real measurement missed the bar. DEGRADED means the pipeline ran and did not produce a measurement.

## Implementation status

| Requirement | Implementation file | Command | Artifact | Status |
| --- | --- | --- | --- | --- |
| Benchmark bundle loader | `cozmo_scan/benchmark.py` | `python run.py benchmark-bundle --manifest benchmarks/properties/not_yet_captured/manifest.json` | structured JSON, exit 0 when evidence is missing and exit 2 when the file is malformed | DONE for the loader. BLOCKED for a real property |
| Laser or tape ground truth | `cozmo_scan/benchmark.py`, `benchmarks/schema/measurement.schema.json` | `python run.py assessment --manifest <path>` | `runs/assessment/<id>/benchmark/ground_truth_validation.json` | DONE for the checker. BLOCKED until a tape file exists |
| Opening gate, ≤2 cm on ≥85% | `score_opening_gate` in `cozmo_scan/benchmark.py` | `python -m pytest tests/test_opening_gate.py` | opening rows in `benchmark.json` | DONE for the scorer. BLOCKED on a real property |
| Ceiling gate, ≤1.5 cm and spread ≤1 cm | `score_ceiling_gate` | `python -m pytest tests/test_ceiling_gate.py` | ceiling rows | DONE for the scorer. BLOCKED on a real property. A FARO match stays BLOCKED |
| Two-capture repeatability | `cozmo_scan/repeatability.py` | `python -m pytest tests/test_repeatability.py` | `repeatability.json` | DONE for two plan files. BLOCKED until capture B exists. The even/odd command is a proxy |
| Photo 2–8 stills, stitch, walls and footprint ≤8% | `cozmo_scan/photo.py`, `score_photo` | `python -m pytest tests/test_photo_gate.py` | property `plan.json` | DONE for the checks. BLOCKED without tape-calibrated stills. The 1.40 m prior is not calibration |
| Video gate, ≤3%, missing pose is DEGRADED | `validate_video_capture`, `score_video_against_ground_truth` | `python -m pytest tests/test_video_gate.py` | video `plan.json` | DONE for the scorer. The company sample is DEGRADED (`no_room_closure`). The ±3% result is BLOCKED without tape |
| Incumbent Polycam or Magicplan, ≥70% beat or tie | `incumbent_bundle`, `score_incumbent` | `python -m pytest tests/test_incumbent.py` | `incumbent.json` | DONE for the importer. BLOCKED until an export of these rooms is supplied |
| Damage fixture, two classes | `cozmo_scan/damage_fixture.py` | `python -m pytest tests/test_damage_fixture.py` | `fixtures/synthetic_damage` | DONE as a synthetic fixture (`source: synthetic_test_fixture`). The physical damage gate is BLOCKED |
| Concealed flag names its rule | `concealed_flags`, `concealed_summary` | `python -m pytest tests/test_damage.py` | `concealed_damage.json` | DONE. An unfired room is `{flag: false, rule: null}`, not a bare boolean |
| Scope line keyed to the surface | `scope_items` | `python -m pytest tests/test_damage.py` | `scope.json` | DONE. A clean room has zero damage scope items |
| Fix loop before, after, compare | `cozmo_scan/fix_loop.py` | `python run.py fix-loop compare --before <dir> --after <dir>` | `diff.json`, `report.md` | DONE for the runner. PARTIAL for the physical gate: one capture is `proxy_experiment: true` |
| Walk-in missing input is JSON | `cozmo_scan/pipeline.py` | `python run.py run --capture <folder> --tier lidar --out <dir>` | stdout JSON | DONE. Missing files exit 0 as BLOCKED. A folder that is not a capture exits 2 |
| Full assessment command | `cozmo_scan/assessment_run.py` | `python run.py assessment --manifest benchmarks/manifests/assessment.example.json` | `runs/assessment/<id>/final_report.md` and the gate table | DONE for the command. The example property is BLOCKED |
| Physical benchmark bundle | `benchmarks/properties/not_yet_captured/` | drop captures in later | tape, repeat walk, incumbent export, staged damage | BLOCKED. The folder is only the layout |

Nothing in this table marks the physical benchmark DONE.

## PDF parts

| PDF part | File | Artifact | Status |
| --- | --- | --- | --- |
| Part 1, Route 2. No TestFlight build | `docs/capture_protocol.md` | one protocol a walk-in can follow | PASS for the page. This machine cannot capture |
| Part 1, three tiers, same JSON contract, wider intervals on thinner sensors | `cozmo_scan/pipeline.py`, `schema/floorplan.schema.json` | `plan.json`, `plan.png` | PASS for the command |
| Part 1, device matrix, including glass, wet floors, low light | `docs/device_matrix.md` | the matrix | PASS for the statement. Centimetre claims are not marked passed |
| Part 2, per-room plan, stitch, damage, concealed flag, scope, confidence interval | `cozmo_scan/layout.py`, `cozmo_scan/damage.py`, `cozmo_scan/contract.py` | `plan.json` | PASS for the contract on a closed room. Physical gates below |
| Part 2, benchmark the candidate builds: ≥3 rooms plus a connector, two damage classes, all three tiers, one repeat, laser or tape | `python run.py assessment --manifest` | `runs/assessment/<id>/` | BLOCKED. Those captures are not in the repo and were not invented |
| Part 2 gates: openings ≤2 cm on ≥85%, ceiling ≤1.5 cm and spread ≤1 cm, repeatability ≤1 cm or 0.5%, photo ±8%, video ±3% | `cozmo_scan/assessment.py` | `benchmark.json` | Scorers verified in `tests/test_gate_local.py`. Physical openings, ceiling, photo, and video are BLOCKED. Repeatability on ARKit visit 421337 is FAIL at 9.7 cm |
| Part 2, drift ablation. Poses used as-is is a fail | `cozmo_scan/fuse.py` anchor on/off | both footprints in `plan.json` | PASS for the ablation row. Loop was checked and not applied |
| Part 3, head-to-head vs Polycam or Magicplan, ≥70% beat or tie | `normalize_incumbent` in `cozmo_scan/assessment.py` | incumbent rows | BLOCKED. No export of these rooms |
| Part 4, fix loop with a regenerable before and after | `docs/fix_loop.md` | even/odd proxy, 0.050 m to 0.0052 m | PASS for that proxy. It is one walk, not two physical captures. The ARKit pair was not “fixed” |
| Part 5, commit history | `git log` | the commits on this branch | Present. Not squashed into one commit |
| Deliverable, technical report, at most 6 pages | `docs/technical_report.md` | the report | PASS for length. It does not claim a laser benchmark |
| Walk-in, cold run on an unseen iPhone 15+ capture | `docs/capture_protocol.md` and `python run.py run` | a plan from the folder they hand over | Ready to run. Not yet scored against their laser |

## Local verification

A blocked physical gate still has a scorer. `python -m pytest tests/test_gate_local.py` runs that scorer on known numbers: PASS, FAIL, the threshold boundary, a missing file, and a malformed file. Those results are not the company benchmark.

| Gate | Real benchmark data | Local fixture | Code | Local verification | Physical benchmark |
| --- | --- | --- | --- | --- | --- |
| Opening scorer | absent | known widths, including 85% and the 2 cm line | `score_openings` | verified | BLOCKED. No tape widths. The fused detector on the synthetic grid is 39/62 |
| Ceiling scorer | absent | known heights, 1.5 cm line, spread, FARO | `score_ceilings` | verified | BLOCKED. FARO peaks miss 1.5 cm and are not tape |
| Repeatability scorer | ARKit visit 421337 | matched walls, the 1 cm line, the 0.5% line, an unmatched wall | `score_repeatability` | verified | FAIL. Long wall 9.7 cm (2.4%) |
| Photo count, scale, wall, footprint | absent | 2 stills, 8 stills, 9 stills rejected, tape scale, 8% line | `validate_photo_capture`, `score_photo` | verified | BLOCKED. No physical stills. The 1.40 m prior cannot pass |
| Video | company sample does not close | synthetic walk builds a cloud and stays `no_room_closure`. A known box laid out with the video tier closes and scores inside 3% | `fuse_video`, `build_layout`, `score_video` | scorer and closed-box layout verified. The walk itself does not close a room | DEGRADED on the company sample and on the synthetic walk. The ±3% gate is BLOCKED without tape |
| Damage scorer | absent | both classes, a false positive, a clean room | `score_damage`, `scope_items` | verified. A synthetic fixture stays BLOCKED even when the classes match | BLOCKED |
| Concealed damage | absent | fired only with the predicate | `concealed_flags` | verified | no physical room required for the rule |
| Scope | absent | a damaged fixture and a clean room | `scope_items` | verified | BLOCKED for a real furnished room |
| Incumbent scorer | absent | 70% beat-or-tie, a tie, a miss, an empty export | `score_incumbent` | verified | BLOCKED. No Polycam or Magicplan export |
| Same property | absent | one property id accepted, a second property rejected | `validate_manifest_identity` | verified | BLOCKED until the bundle is present |
| Tape schema | absent | complete tape row, bad source, wrong unit, incomplete row, FARO | `measurement.schema.json` | verified | BLOCKED. FARO stays a reference |

Exit codes on these fixtures: FAIL exits 1, a mixed property exits 2, and a video that does not close stays DEGRADED with exit 0. DEGRADED is not reported as PASS.

| PDF requirement | Implementation | Input | Output | Threshold | Evidence | Status | Reproduction command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| One command writes schema JSON, a rendered plan, and a confidence interval on every length | `run_one` in `cozmo_scan/pipeline.py` | one capture folder and a tier | `runs/<name>/plan.json`, `plan.png` | schema valid | `runs/single_room_lidar/plan.json` when the company sample is present | PASS for the command | `python run.py run --capture single_room/c00a170fe1 --tier lidar --out runs/single_room_lidar` |
| Room plan: walls, wall lengths, ceiling, floor area, openings, opening widths, confidence interval | `build_layout` in `cozmo_scan/layout.py`, `apply_contract` | a capture that closes a room | `plan.json` rooms | every measured length has value, sigma, ci95 | self-check box 4.00 x 3.00 x 2.50 m, door 0.863 m | PASS for the contract on the synthetic box. Company openings have no tape | `python run.py self-check` |
| Property stitch: rooms, polygons, adjacency, connector, no accidental overlap, footprint | `stitch_rooms` in `cozmo_scan/photo.py` | per-room photo folders | property plan | shared edge only; overlap rejected | synthetic fixture, 4 rooms, overlap 0 | BLOCKED as a physical result | `python run.py run --capture fixtures/synthetic_photo_property --tier photo --out runs/synthetic_photo_property` |
| Damage: surface, class, region, metric extent, confidence | `detect_on_points` in `cozmo_scan/damage.py` | furnished room, two staged classes | `damage.json` | class and extent on that room | `fixtures/synthetic_damage` only | BLOCKED | `python -m pytest tests/test_damage.py` |
| Concealed flag: rule_id, surface, trigger, evidence, confidence, fired. Missing evidence does not fire | `concealed_flags` | a plan with or without a stain | `concealed_damage` | fired only when the predicate holds | unfired when the predicate is absent | PASS for the rule engine | `python -m pytest tests/test_damage.py` |
| Scope line: room, surface, category, quantity, unit, evidence, confidence. A clean room has none | `scope_items` | detected damage, or none | `scope.json` | one line per damage item | synthetic fixture has stain and moisture; an undamaged plan has `[]` | PASS for the mapping. The physical room is BLOCKED | `python -m pytest tests/test_damage.py` |
| Opening width | `score_openings`; per-frame API `detect_openings_multiframe`. Shipped LiDAR is still `voxel_fused` | tape widths plus predictions with a confidence interval | `benchmark.json` openings | ≤2 cm on ≥85%. A miss and a phantom each count | fused synthetic grid 39/62 = 62.9%. Five synthetic frames 62/62. No tape | BLOCKED | `python run.py assessment --manifest benchmarks/manifests/assessment.example.json` |
| Ceiling height and repeat spread | `score_ceilings` | tape ceiling per room, and a second walk for the spread | ceiling rows in `benchmark.json` | ≤1.5 cm, spread ≤1 cm | phone vs FARO peak 5.4 cm and 4.5 cm. Spread 1.1 cm. FARO is not tape | BLOCKED | `python -m datasets.arkitscenes_reference` |
| Repeatability, unmatched walls listed | `score_repeatability` | the same room and tier, captured twice | matched, unmatched, failed, max absolute, max relative | ≤1 cm or ≤0.5% | ARKit visit 421337 long wall 9.7 cm (2.4%) | FAIL on that pair. A tape protocol is still absent | both ARKit LiDAR commands in `docs/benchmark_report.md` |
| Drift ablation. Poses used as-is is not a pass | `footprint_area_m2_anchor_on/off` | one LiDAR walk | both areas in the plan | report on and off | 14.000 m² vs 10.965 m². Loop checked, not applied | PASS for the ablation row | LiDAR command above |
| Photo walls | `score_photo`, `validate_photo_capture` | 2–8 stills per room and a tape scale | photo wall gate | ≤8% | 1.40 m prior is estimated | BLOCKED | photo command above |
| Photo footprint | `score_photo` | the same stills and a tape footprint | photo footprint gate | ≤8% | synthetic footprint only | BLOCKED | photo command above |
| Photo count and no mixed rooms | `validate_photo_capture` | `room_01/` … stills | `validation.json` | 2–8 stills, one property | 9 stills are rejected by the assessment command. The older photo command still caps a 9th still at 8 | PASS for the check. Physical stills are absent | `python -m pytest tests/test_assessment.py` |
| Video walls. No invented room when closure fails | `fuse_video`, `score_video` | ordered mp4, timestamps if odometry exists | video gate | ≤3% when the room closes | company video `no_room_closure`, rooms 0 | DEGRADED reconstruction. The ±3% gate is BLOCKED without tape | `python run.py run --capture single_room/c00a170fe1 --tier video --out runs/single_room_video` |
| Incumbent, same rooms | `score_incumbent` | Polycam or Magicplan export of this property | ours, incumbent, tape, both errors, winner | ≥70% beat or tie | no export in the repo | BLOCKED | `python run.py assessment --manifest benchmarks/manifests/assessment.example.json` |
| Same property across LiDAR, video, photos, repeat, tape, incumbent | `validate_manifest_identity` | one manifest | `validation.json` | mismatched property_id or room is rejected | example manifest paths are empty | BLOCKED until the company bundle is dropped in | `python run.py assessment --manifest benchmarks/manifests/assessment.example.json` |
| LiDAR capture contract: depth, confidence, intrinsics, odometry. RGB optional | `validate_lidar_dir` | iPhone depth bundle | `validation.json` | missing file is an error, not a silent video fallback | checked in `tests/test_assessment.py` | PASS for the check | `python -m pytest tests/test_assessment.py` |
| Ground truth schema. Tape, laser, or survey only for the physical gates | `measurement_contract_issues`, `benchmarks/schema/measurement.schema.json` | measurements with property, room, value, unit, uncertainty, source, capture_id | schema errors in `validation.json` | FARO stays a reference | no tape file in the repo | BLOCKED | `python run.py assessment --manifest benchmarks/manifests/assessment.example.json` |
| Reproducibility: commit, command, Python, packages, platform, timestamp, hashes | `run_assessment` | the manifest | `environment.json`, `command.txt` | no API keys, relative paths in the report | written on every assessment run | PASS for the record | assessment command above |
| States PASS, FAIL, BLOCKED, DEGRADED. DEGRADED is not turned into PASS | `score_*` and `run_assessment` | a manifest | `final_report.md` | exit 1 only on FAIL. Mixed properties exit 2 | example run is BLOCKED | PASS for the state machine. Physical gates are not passed | `python -m pytest tests/test_assessment.py` |
| Fix loop records the worst remaining gate without calling a synthetic gain a real fix | `docs/fix_loop.md` | benchmark notes | this file | do not retune thresholds | opening and ceiling sections say the physical gates are not closed | PASS for the write-up | read `docs/fix_loop.md` |
