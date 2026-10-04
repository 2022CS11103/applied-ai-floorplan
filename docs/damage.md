# Damage

Physical damage-detection accuracy is not validated. There is no staged real capture and no tape or survey of a real stain. The sample walks and the photo property fixture stay undamaged. The detector does not paint them.

## Classes

The detector emits two classes. Both are names it already used.

| Class | What the color has to do | Scope code |
| --- | --- | --- |
| `stain` | A compact patch darker than the wall around it | `INTERIOR_PAINT` |
| `moisture` | That same darkening, plus a yellow shift, with the patch's top at or below 0.45 m | `MOISTURE_OPEN_AND_INSPECT` |

`discoloration` is accepted by the scope catalog and the schema. This detector does not emit it. A class probability is not estimated. `confidence` is null.

## Input

Colored points in the same horizontal frame as the room walls, plus the wall segment (`p0_m`, `p1_m`). LiDAR builds those colors with a second pass over the depth frames. Video uses colors from the triangulation when they exist. A photo plan does not run this detector. The synthetic photo property has no damage list because nothing was painted on it.

## Detection

`cozmo_scan/damage.py` bins each wall into 10 cm by 10 cm cells and keeps the median color. A cell counts when it differs from the median of its neighbors by at least 32 OpenCV LAB units and is darker by at least 12. A lighting gradient moves the neighbors too, so it does not fire. A region needs eight such cells, at least 0.12 m², and a compact footprint. The area is the cell count times 0.01 m².

## Extent

The metric is area, in square metres, on `extent_m2`. The field `extent` is that same object. It has `value`, `sigma`, `ci95_low`, and `ci95_high`. Sigma starts at 40% of that area, and at least 0.01 m², then the tier floor widens the interval (1.5 cm absolute on LiDAR, 3 cm and 3% on video). The region also stores the height band, the along-wall span, and the centroid. Those describe where the cells sit. They are not a second uncertain measurement.

## Concealed damage

The capture does not see behind a wall. Two rules interpret the visible patch. Each wall records whether the rule fired, including when it did not.

| Rule | Fires when |
| --- | --- |
| `base_stain_moisture_path` | A stain or moisture region starts at or below 0.30 m and covers at least 0.04 m². |
| `sill_stain_under_window` | The wall has a window and a stain or moisture region reaches 0.70 m. |

A region with one of those rules has `concealed: true` and lists the rule. That flag means the visible evidence matches the rule. `evidence.concealed_observed_directly` is false.

## Synthetic fixture

`fixtures/synthetic_damage` is a synthetic test fixture, not physical benchmark ground truth.

| Path | Role |
| --- | --- |
| `input/scene.json` | One room, a sofa off the wall, a dark band, and a yellow band at the base. |
| `ground_truth.json` | The painted rectangle areas. Not the detector output. |
| `FIXTURE.txt` | The label. |

The sofa is in the cloud so the room is furnished. It does not lie on a wall, so it is not a damage region. A second wall is unpainted and stays clean.

## Limitations

The residual is hand-tuned on LAB units, not trained, and not checked on a real stain. A dark picture frame or a poster can look like a stain. A thin band is what the neighbor test can see. A large uniform patch whose neighbors are the same color does not fire, because the test is local. Cell area is coarser than a tape. Do not read a fixture extent as assessment accuracy.
