# Floor plans from a phone capture

One command turns a capture folder into a dimensioned plan, a JSON file, and a confidence interval on every length. Three tiers share that contract: LiDAR, video, and photos. Thinner sensors get wider intervals. A ceiling that was not scanned is reported as not observed.

```bash
pip install -r requirements.txt
python run.py self-check
python run.py assessment-gate --manifest benchmarks/manifests/example.json
python run.py assessment --manifest benchmarks/manifests/assessment.example.json
python run.py run --capture single_room/c00a170fe1 --tier lidar --out runs/single_room_lidar
```

`self-check` builds a 4.00 m by 3.00 m by 2.50 m box with a door and checks the plan. It does not need the sample data. A LiDAR run on the supplied room finishes in about 20 seconds.

```bash
pytest
```

`pytest` reruns that synthetic check and, when `single_room/c00a170fe1` is on disk, the LiDAR command. The area check locks the current output. It is not a laser measurement.

## Sample data

The company sample is local and is not in git (the depth images and video are hundreds of megabytes):

- `single_room/c00a170fe1`
- `single_scan_with_ceiling/c7d28f72c6`
- `single_scan_floor_only/1a8384c3f6` (same bundle shape)

```bash
python run.py run --capture single_scan_with_ceiling/c7d28f72c6 --tier lidar --out runs/ceiling_lidar
python run.py run --capture single_room/c00a170fe1 --tier video --out runs/single_room_video
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_after --endpoints intersections
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_before --endpoints observed
```

Video on a LiDAR bundle ignores the depth images and uses the metric poses. Frames are read in order, because seeking the mp4 often returns the wrong picture. On this sample the cloud is denser than the old few-hundred-point triangulation, and the plan stays degraded when those points still do not close a room. The JSON then carries `degraded_reasons`. That refusal is the result. The video wall-length gate stays blocked without a tape.

Photos are one folder of 2–8 stills per room. A property is the parent of those folders. Rooms are stitched only where the floor polygons share an edge; a handwritten adjacency file is ignored, and overlapping floors are rejected. There is no iPhone 15+ photo capture in this repo, so the checked property is `fixtures/synthetic_photo_property`. That folder is a synthetic test fixture, not physical ground truth. The photo gates (footprint ±8%, wall lengths ±8%, adjacency on a real capture) stay blocked without a tape.

Damage is a local color residual on a wall: `stain` or `moisture`, with area in square metres. A concealed flag is a written rule about that visible patch, not a view behind the wall. The only staged example is `fixtures/synthetic_damage`, and it is a synthetic test fixture, not physical ground truth. See `docs/damage.md`.

The accuracy harness reads a `plan.json` and a separate ground-truth file. It does not rebuild the plan. Sample captures have no laser or tape file, so their report stays blocked. The cases under `benchmarks/fixtures/` only check the arithmetic. See `docs/benchmark.md`.

## Assessment bundle

When the company's captures and tape file are available, one command validates them and scores every gate:

```bash
python run.py assessment --manifest path/to/manifest.json
```

The manifest names one property. LiDAR, video, photos, the repeat walk, the tape file, the damage sheet, and a Polycam or Magicplan export must all carry that same `property_id`. A second property in the same manifest is rejected before any plan is built. The folder the manifest describes looks like this:

```text
assessment/
    property/lidar/          depth, confidence, odometry.csv, camera_matrix.csv
    property/video/          walkthrough.mp4, timestamps in order
    property/photos/room_01/ 2 to 8 stills, and the same for room_02 and room_03
    ground_truth/measurements.json
    incumbent/polycam.json
    repeats/
```

`benchmarks/manifests/assessment.example.json` is that shape with nothing filled in. Running it is BLOCKED, not a pass. Output lands in `runs/assessment/<property_id>/` as `validation.json`, one directory per tier, `benchmark.json`, `compliance.json`, `environment.json`, and `final_report.md`.

A gate is PASS only when a tape, laser, or survey measurement exists and meets the bar. FAIL means the measurement exists and misses the bar. BLOCKED means the physical file is missing, the source is FARO or another external reference, or the photo scale is still the 1.40 m prior. DEGRADED means that tier ran and did not close a room; the command does not invent the missing lengths. The synthetic 62-opening schedule and the synthetic photo property stay in the tests. They are not this command's result.

Reproduce the other checks the same way:

```bash
python run.py self-check
python -m pytest -q
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_after --endpoints intersections
python -m datasets.arkitscenes_reference
```

The repeat command on one company walk is an even/odd proxy. It is not the two-walk repeatability gate. ARKitScenes is a FARO reference, not a tape.

## Output

`runs/<name>/plan.json` follows `schema/floorplan.schema.json`. `plan.png` is the drawing. Lengths are metres. The room graph is the same on every tier: `room_id` and `floor_polygon` on each room, and adjacency as `room_a`, `room_b`, `relationship`, `source`, and `shared_length_m` only when the floor polygons already share an edge. Every measured length keeps `value`, `sigma`, `ci95_low`, and `ci95_high`. A ceiling that was not seen is `null`, not a guessed interval. `degraded_reasons` is always present. An empty room list is never `status=ok`. Details are in `docs/plan_contract.md`.

## Docs

| | |
| --- | --- |
| Capture route and device matrix | `docs/capture_protocol.md`, `docs/device_matrix.md` |
| What passed, what did not | `docs/benchmark_report.md` |
| The fix that was shipped | `docs/fix_loop.md` |
| How it works | `docs/technical_report.md` |
| Requirement to file | `docs/compliance_matrix.md` |

## Why the numbers look the way they do

The logger stores an OpenCV camera (X right, Y down, Z forward) and a camera-to-world quaternion. In that frame Y is up. Using an ARKit camera axis on this file stretches the vertical span to about 5 m, which is not a room. Walls are the columns that are tall, not every return: a table is a thin slab. Room length is the distance between the cross walls, not the stretch of wall that one pass happened to see. That last choice is the fix in `docs/fix_loop.md`.
