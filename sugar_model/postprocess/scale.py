"""(e) Metric scaling.

SuGaR/COLMAP coordinates have an arbitrary scale.  Three ways to fix it
(`--scale-ref`):

* `points=x1,y1,z1;x2,y2,z2;D` - two points picked on the RAW mesh (e.g. the
  two ends of a tape measure / ArUco board edge, picked in MeshLab or
  CloudCompare) and their true distance D in --units.
* `counter_height=H` - floor-to-countertop-top distance measured with a tape
  (standard US base: 36 in = 914.4 mm).  Measured automatically from the two
  dominant up-facing horizontal planes, so it is robust (thousands of points
  vote).  This is the default (H = 36 in) when nothing else is given.
* `factor=s` - multiply raw coordinates by s.

ArUco/AprilTag scale (not implemented here, documented in README): detect the
board corners in >= 3 undistorted frames with cv2.aruco, triangulate them with
the COLMAP poses, and use the known board edge length as a `points=` ref.
"""
from __future__ import annotations

import numpy as np

from .common import log


def parse_scale_ref(s: str | None):
    if not s:
        return ("counter_height", 914.4 / 25.4, "in", True)
    kind, _, val = s.partition("=")
    kind = kind.strip().lower()
    if kind == "points":
        a, b, d = val.split(";")
        return ("points", (np.array(list(map(float, a.split(",")))),
                           np.array(list(map(float, b.split(",")))), float(d)), None, False)
    if kind == "counter_height":
        return ("counter_height", float(val), None, False)
    if kind == "factor":
        return ("factor", float(val), None, False)
    raise ValueError(f"unknown --scale-ref '{s}'")


def scale_from_points(p1, p2, dist):
    return dist / float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def measure_counter_height(V, normals, areas, centroids, rel_band=(0.2, 0.6)):
    """In a Z-up aligned (but unscaled) mesh, return (floor_z, counter_z).

    Floor  = lowest strong peak of up-facing area along Z.
    Counter = strongest up-facing peak whose relative height (z-floor)/(zmax-floor)
              is inside rel_band (36 in / 96 in ceiling = 0.375)."""
    up = normals[:, 2] > np.cos(np.radians(10))
    z, w = centroids[up, 2], areas[up]
    zmin, zmax = V[:, 2].min(), V[:, 2].max()
    nb = 1000
    hist, edges = np.histogram(z, bins=nb, range=(zmin, zmax), weights=w)
    k = 3
    sm = np.convolve(hist, np.ones(2 * k + 1), "same")
    peaks = [i for i in range(1, nb - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0]
    tot = sm.sum() / (2 * k + 1)

    def refine(i):
        lo, hi = edges[max(i - 2 * k, 0)], edges[min(i + 2 * k + 1, nb)]
        s = (z >= lo) & (z <= hi)
        return float(np.average(z[s], weights=w[s]))

    strong = [i for i in peaks if sm[i] / (2 * k + 1) > 0.05 * tot]
    floor_i = min(strong)
    floor_z = refine(floor_i)
    cand = [i for i in peaks
            if rel_band[0] < (edges[i] - floor_z) / (zmax - floor_z) < rel_band[1]]
    if not cand:
        raise RuntimeError("could not find a countertop-height plane; pass --scale-ref points=... or factor=...")
    ci = max(cand, key=lambda i: sm[i])
    counter_z = refine(ci)
    log.info("counter-height scale: floor z=%.5f counter z=%.5f (raw units)", floor_z, counter_z)
    return floor_z, counter_z
