"""A dimensioned plan a homeowner can compare to a scanning app.

Coordinates are the capture's floor plane (x, z), not compass north.
The logger does not record a heading.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_plan(document: dict, path: Path) -> None:
    path = Path(path)
    rooms = document["property"]["rooms"]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f5f1")
    fig.patch.set_facecolor("#f7f5f1")

    if not rooms:
        ax.text(0.5, 0.5, "No closed room in this capture", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    all_pts = []
    for room in rooms:
        poly = np.array(room["polygon_m"], dtype=float)
        all_pts.append(poly)
        ax.fill(poly[:, 0], poly[:, 1], color="#e7eef6", zorder=1)
        ax.plot(
            np.r_[poly[:, 0], poly[0, 0]],
            np.r_[poly[:, 1], poly[0, 1]],
            color="#1c1c1c",
            lw=2.2,
            zorder=3,
        )
        c = poly.mean(axis=0)
        area = room["floor_area_m2"]["value"]
        ceil = room.get("ceiling_height_m")
        ceil_txt = f"\nceiling {ceil['value']:.2f} m" if ceil else "\nceiling not observed"
        ax.text(
            c[0],
            c[1],
            f"{room['id'].replace('_', ' ')}\n{area:.2f} m²{ceil_txt}",
            ha="center",
            va="center",
            fontsize=8,
            color="#1d3557",
            zorder=4,
        )
        for wall in room["walls"]:
            p0 = np.array(wall["p0_m"])
            p1 = np.array(wall["p1_m"])
            _dimension(ax, p0, p1, wall["length_m"]["value"])
            for op in wall["openings"]:
                _opening(ax, p0, p1, op)

        for dmg in room.get("damage", []):
            cx, cz = dmg["centroid_m"]
            ax.scatter([cx], [cz], s=36, c="#b42318", zorder=5)
            ax.text(cx, cz, f"  {dmg['class']}", color="#b42318", fontsize=7, zorder=5)

    pts = np.concatenate(all_pts)
    pad = 0.8
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    _scale_bar(ax)
    title = f"{document['capture_id']}  ·  {document['tier']}  ·  {document.get('status', 'ok')}"
    ax.set_title(title, loc="left", fontsize=11, color="#222")
    ax.set_xlabel("plan x (m)")
    ax.set_ylabel("plan z (m)")
    ax.grid(True, color="#ddd", lw=0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _write_svg(document, path.with_suffix(".svg"))


def _dimension(ax, p0, p1, length):
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 0.2:
        return
    direction = d / L
    normal = np.array([-direction[1], direction[0]])
    # push the dimension line outside the room: caller doesn't know inside,
    # so offset a fixed way and keep the text readable
    off = normal * 0.28
    a, b = p0 + off, p1 + off
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(arrowstyle="<->", color="#3d5a80", lw=0.8),
        zorder=4,
    )
    mid = 0.5 * (a + b)
    ax.text(mid[0], mid[1], f"{length:.2f} m", color="#3d5a80", fontsize=7, ha="center", va="bottom", zorder=4)


def _opening(ax, p0, p1, opening):
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 0.2:
        return
    direction = d / L
    width = opening["width_m"]["value"]
    # the along-track of the gap is not stored in metres from p0; draw the opening
    # centered, which is honest about width and does not pretend we know the hinge side
    mid = 0.5 * (p0 + p1)
    a = mid - direction * (width / 2)
    b = mid + direction * (width / 2)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#f7f5f1", lw=4, zorder=3, solid_capstyle="butt")
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#c05621", lw=1.4, zorder=4)
    klass = opening["class"]
    ax.text(mid[0], mid[1], f"{klass[0].upper()} {width:.2f}", color="#c05621", fontsize=6.5, ha="center", va="top")


def _scale_bar(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = 1.0
    start = np.array([x0 + 0.15 * (x1 - x0), y0 + 0.08 * (y1 - y0)])
    end = start + np.array([length, 0])
    ax.plot([start[0], end[0]], [start[1], end[1]], color="#111", lw=2, zorder=5)
    ax.plot([start[0], start[0]], [start[1] - 0.04, start[1] + 0.04], color="#111", lw=1.2)
    ax.plot([end[0], end[0]], [end[1] - 0.04, end[1] + 0.04], color="#111", lw=1.2)
    ax.text((start[0] + end[0]) / 2, start[1] + 0.06, "1 m", ha="center", fontsize=7)


def _write_svg(document: dict, path: Path) -> None:
    rooms = document["property"]["rooms"]
    polys = [np.array(r["polygon_m"], dtype=float) for r in rooms if r["polygon_m"]]
    if not polys:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        return
    allp = np.concatenate(polys)
    minx, miny = allp.min(0) - 1
    maxx, maxy = allp.max(0) + 1
    W, H = 900, 700
    sx = W / max(maxx - minx, 0.1)
    sy = H / max(maxy - miny, 0.1)
    s = min(sx, sy)

    def xy(p):
        return (p[0] - minx) * s, H - (p[1] - miny) * s

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>", "<rect width='100%' height='100%' fill='#f7f5f1'/>"]
    for room in rooms:
        poly = np.array(room["polygon_m"], dtype=float)
        pts = " ".join(f"{xy(p)[0]:.1f},{xy(p)[1]:.1f}" for p in poly)
        parts.append(f"<polygon points='{pts}' fill='#e7eef6' stroke='#1c1c1c' stroke-width='2'/>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
