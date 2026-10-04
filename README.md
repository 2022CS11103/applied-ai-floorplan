# Floor plans from a phone capture

One command turns a capture folder into a dimensioned plan, a JSON file, and a confidence interval on every length. Three tiers share that contract: LiDAR, video, and photos. Thinner sensors get wider intervals. A ceiling that was not scanned is reported as not observed.

```bash
pip install -r requirements.txt
python run.py self-check
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
