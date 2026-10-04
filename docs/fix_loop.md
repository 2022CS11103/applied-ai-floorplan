# Fix loop

Capture: `single_room/c00a170fe1`, LiDAR tier.
Proxy, not two physical walks: even frames against odd frames of the same walk. A second walk was not possible. The proxy is labelled in `repeatability.json`.

## 1. Worst gate

Repeatability. Two passes of the same room should agree within 1 cm or 0.5% on a wall, whichever is larger.

Before the fix, the worst matched wall moved by **0.05 m**. Its tolerance was **0.0177 m**. One of the two matched walls passed, one failed. Gate failed.

## 2. Root cause

The wall length was the stretch of surface those frames happened to see.

Evidence, from `runs/repeat_before/repeatability.json`:

| Pass | Length of the same wall |
| --- | --- |
| even frames | 3.55 m |
| odd frames | 3.60 m |

The midpoints were 11 cm apart, so it is the same wall. The 5 cm gap is the difference in how far along that wall each subset got returns, not a change in the room. The full before plan (`runs/single_room_before`) shows the same mistake at capture scale: observed runs of 4.65 m on a room whose other side is 2.40 m.

## 3. The fix, and the number it was aimed at

Report the distance between the two cross-walls that close the room. Both subsets still see those cross-walls even when they see different portions of the long wall. Predicted worst delta after the fix: under 1 cm.

Shipped as the default `--endpoints intersections`. The old behaviour remains `--endpoints observed`, so the before run is regenerable.

## 4. After

`runs/repeat_after/repeatability.json`:

| | Before (`observed`) | After (`intersections`) |
| --- | --- | --- |
| Walls compared | 2 | 2 |
| Inside the gate | 1 | 2 |
| Worst absolute delta | 0.050 m | 0.0052 m |
| Gate | fail | pass on the walls that matched |

0.52 cm is inside the 1.2 cm tolerance on a 2.40 m wall. The prediction (under 1 cm) was in the right direction and inside the gate.

What did not get fixed: even frames still close 3 rooms and odd frames close 2. The split agrees on the walls it can match. It does not agree on how many rooms the furniture ridges produce. That is a coverage and clutter problem, and it is still open. The report says so rather than calling the whole repeatability row solved for a second physical capture.

## Regenerate

```bash
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_before --endpoints observed
python run.py repeat --capture single_room/c00a170fe1 --out runs/repeat_after --endpoints intersections
python run.py run --capture single_room/c00a170fe1 --tier lidar --endpoints observed --out runs/single_room_before
python run.py run --capture single_room/c00a170fe1 --tier lidar --endpoints intersections --out runs/single_room_lidar
```
