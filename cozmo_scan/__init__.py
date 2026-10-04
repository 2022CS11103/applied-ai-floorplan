"""Property floor plans from handheld phone captures.

Three input tiers share one output contract:

* lidar — depth, confidence, metric poses
* video — RGB walkthrough, metric if VIO poses exist, otherwise scaled SfM
* photo — 2 to 8 stills per room, scaled by the capture-protocol camera height
"""

__version__ = "1.0.0"
