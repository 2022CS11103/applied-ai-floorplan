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

Video on a LiDAR bundle ignores the depth images. On this sample it triangulates a few hundred points and refuses to draw a room. That refusal is the result.

## Output

`runs/<name>/plan.json` follows `schema/floorplan.schema.json`. `plan.png` is the drawing. Lengths are metres.

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
