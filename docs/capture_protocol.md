# Capture protocol

Route 2. There is no TestFlight app in this submission. The author does not have a working LiDAR phone, and the brief says a stock protocol is a legitimate route. At a walk-in, follow this page literally.

The pipeline does not talk to a phone. It reads a folder.

## What to install

LiDAR tier, iPhone 15 Pro or any later Pro:

1. Install **SiteScape** or any logger that writes the bundle below. The sample data already is that bundle. If the walk-in phone cannot install a logger, use the Record3D or Stray Scanner style export and rename files to match the table. Do not convert units.
2. Photos tier, any iPhone 15 or newer: the built-in Camera app. No extra install.
3. Video tier, any iPhone 15 or newer: the built-in Camera app, video mode, 1080p or 4K. Higher is not better here.

## The bundle the LiDAR command expects

```
<capture_id>/
    camera_matrix.csv      3x3 intrinsics of the RGB camera
    odometry.csv           one pose per RGB frame, metres, quaternion xyzw
    imu.csv                accelerometer in g, gyro in rad/s
    rgb.mp4                frames aligned with the odometry frame index
    depth/000000.png       uint16 millimetres, 256x192
    confidence/000000.png  0, 1, or 2
```

`python run.py run --capture <capture_id> --tier lidar --out runs/walkin_lidar`

## How to walk (LiDAR)

1. Start in the doorway, phone at chest height, lens roughly level. Chest height is about 1.4 m. This number is the photo-tier scale prior, so do not hold it overhead.
2. Walk the perimeter once, slowly, pointing the phone at the walls, not at the floor and not at the ceiling the whole time. A few seconds of ceiling, then back to the walls.
3. Pass through every doorway. Stand in the doorway and turn to face each room.
4. One room takes about 45 to 90 seconds. A three-room property takes about 4 minutes. Do not rush and do not rescan the same wall five times.
5. Avoid mirrors, glass, and wet floors as the surface you are measuring. They will be in the room; do not aim at them for long. Low light is acceptable. A dark room widens the interval, it does not need a lamp in the export.
6. Stop the log. AirDrop or copy the folder as-is. Do not zip only the video.

## How to walk (video)

1. Same walk as above, built-in Camera, 30 fps is enough.
2. Hold the phone in landscape or portrait, but do not switch mid-walk.
3. One clip per property, not one clip per room.
4. Hand over the mp4. If the phone also logged poses and IMU, put `odometry.csv` and `imu.csv` next to the file. Without poses the video tier runs structure-from-motion and scales it with the 1.40 m chest-height prior. The interval gets wider. That is expected.

`python run.py run --capture <folder-with-mp4> --tier video --out runs/walkin_video`

## How to shoot (photos)

1. Two to eight stills per room. One folder per room.
2. Stand in a corner, chest height, phone upright, and photograph the opposite corner so two walls and some floor are in frame. Then the other corners. Overlap is required. A set of photos of one wall only will not make a plan.
3. Take one extra photo standing in each doorway, looking into the next room.
4. Write `adjacency.json` next to the room folders. This is the sketch. The photo tier cannot invent which room is north of which room from eight stills.

```json
{
  "rooms": [
    {"id": "kitchen", "folder": "kitchen"},
    {"id": "hall", "folder": "hall"}
  ],
  "links": [
    {"a": "kitchen", "b": "hall", "via": "door", "side": "north"}
  ]
}
```

`side` is the side of room `a` that room `b` sits on: north, south, east, or west, in the plan drawing, not a compass.

`python run.py run --capture <property-folder> --tier photo --out runs/walkin_photo`

## What to avoid

- Do not measure with the phone on a gimbal or a tripod. The pipeline assumes a walking capture.
- Do not crop the video.
- Do not run a beauty filter or a depth-blur portrait mode.
- Do not delete `confidence/` to save space. Low-confidence returns are how glass is rejected.

## What you get back

One JSON file and one PNG plan per command. Measurements are metres. Every length has a 95% interval. A missing ceiling is reported as not observed, not as a guessed 2.4 m.
