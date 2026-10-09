#!/usr/bin/env python3
"""End-to-end test of postprocess/cad_clean.py on a synthetic SuGaR-like kitchen.

Run:  python3 tests/test_cad_clean.py [--keep-dir DIR] [--seed N]
(no pytest needed; also works under pytest: `pytest tests/test_cad_clean.py`)

The test maps every recovered plane back into the ground-truth frame using the
transform the tool writes to its report (raw->output) and the known
truth->raw transform of the generator, then checks:

  1. floaters removed, detached bar pull kept
  2. recovered plane normals within 0.5 deg of truth
  3. orthogonal plane pairs at 90 +/- 0.5 deg (in the output frame)
  4. inlier RMS < 1 mm after snapping: RMS distance of the output vertices of
     each plane to the TRUE plane (mm), for every plane >= 0.05 m^2, plus the
     area-weighted RMS over ALL matched planes.  (Faces smaller than the ripple
     wavelength cannot average a 3 mm low-frequency ripple away; they are
     reported, not asserted.)  RMS of output vertices to their OWN snapped
     plane is also reported (should be ~0: every patch is exactly flat).
  5. metric scale: distances between parallel truth planes reproduced < 2 mm
  6. face count reduced by >= 80 %
  7. output is watertight, manifold, consistently wound, no degenerate faces
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_synthetic import make_synthetic  # noqa: E402

TOL_ANGLE = 0.5      # deg
TOL_RMS = 1.0        # mm
MIN_AREA_RMS = 0.05e6  # mm^2, planes used for the RMS / metric assertions (>= 500 cm^2)


def run(keep_dir=None, seed=7):
    d = make_synthetic(seed=seed)
    work = keep_dir or tempfile.mkdtemp(prefix="cadclean_")
    os.makedirs(work, exist_ok=True)
    raw_path = os.path.join(work, "synthetic_kitchen.ply")
    d["raw"].export(raw_path)
    out_dir = os.path.join(work, "out")
    cmd = [sys.executable, os.path.join(ROOT, "postprocess", "cad_clean.py"), raw_path, out_dir,
           "--units", "mm", "--scale-ref", "counter_height=914.4"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise SystemExit("cad_clean failed")
    rep = json.load(open(os.path.join(out_dir, "synthetic_kitchen_report.json")))
    out = trimesh.load(os.path.join(out_dir, "synthetic_kitchen_clean.ply"), process=False)
    fp = np.load(os.path.join(out_dir, "synthetic_kitchen_face_plane_ids.npy"))
    V, F = np.asarray(out.vertices, float), np.asarray(out.faces)

    # output -> truth(mm):  out = M raw,  raw = T truth
    M = np.array(rep["transform_raw_to_output"])
    T = d["T_truth_to_raw"]
    O2T = np.linalg.inv(M @ T)
    Vt = V @ O2T[:3, :3].T + O2T[:3, 3]
    Rot = O2T[:3, :3] / np.cbrt(np.linalg.det(O2T[:3, :3]))   # pure rotation part

    results = {"runtime_s": round(runtime, 2), "checks": {}}
    ok_all = True

    def check(name, cond, detail):
        nonlocal ok_all
        results["checks"][name] = {"pass": bool(cond), **detail}
        ok_all &= bool(cond)

    # 1. floaters
    tree_d = []
    from scipy.spatial import cKDTree
    tree = cKDTree(Vt)
    for f in d["floaters"]:
        tree_d.append(float(tree.query(f["center"])[0]))
    min_clear = min(tree_d)
    det_d = float(tree.query(d["detached_handle_center"])[0])
    check("floaters_removed", min_clear > 100 and rep["floaters"]["faces_removed"] > 0,
          {"n_floaters": len(d["floaters"]),
           "components_in": rep["floaters"]["components_in"],
           "components_kept": rep["floaters"]["components_kept"],
           "removed": rep["floaters"]["removed"],
           "min_distance_output_to_any_floater_center_mm": round(min_clear, 1)})
    check("detached_handle_kept", det_d < 15,
          {"distance_mm": round(det_d, 2),
           "hardware_faces": rep["layers"].get("hardware", {}).get("faces", 0)})

    # 2 + 4. per plane: match to truth, normal angle, RMS to true plane
    truth = d["truth_planes"]
    rows = []
    for p in rep["planes"]:
        i = p["id"]
        fs = fp == i
        if not fs.any():
            continue
        n_t = Rot @ np.array(p["normal"])
        vid = np.unique(F[fs])
        P = Vt[vid]
        best = None
        for tn, td, tname in truth:
            if tn @ n_t < np.cos(np.radians(5)):
                continue
            dist = np.abs(P @ tn - td)
            r = float(np.sqrt(np.mean(dist ** 2)))
            if best is None or r < best[0]:
                best = (r, tname, tn, td)
        if best is None or best[0] > 10:
            rows.append({"id": i, "match": None, "area_m2": p["area"] / 1e6})
            continue
        ang = float(np.degrees(np.arccos(np.clip(best[2] @ n_t, -1, 1))))
        rows.append({"id": i, "match": best[1], "area_m2": round(p["area"] / 1e6, 4),
                     "angle_err_deg": round(ang, 4), "rms_to_truth_mm": round(best[0], 4),
                     "rms_fit_before_snap_mm": p["inlier_rms_fit_before_snap"],
                     "rms_final_to_own_plane_mm": p["inlier_rms_final"],
                     "snapped": p["snapped_axis"], "layers": p["layers"]})
    matched = [r for r in rows if r["match"]]
    big = [r for r in matched if r["area_m2"] * 1e6 >= MIN_AREA_RMS]
    max_ang = max(r["angle_err_deg"] for r in matched)
    check("normals_within_0.5deg", max_ang < TOL_ANGLE,
          {"planes_matched": len(matched), "planes_total": len(rows),
           "max_angle_err_deg": round(max_ang, 4),
           "mean_angle_err_deg": round(float(np.mean([r["angle_err_deg"] for r in matched])), 4)})
    max_rms = max(r["rms_to_truth_mm"] for r in big)
    area_w = np.array([r["area_m2"] for r in matched])
    aw_rms = float(np.sqrt(np.average(np.array([r["rms_to_truth_mm"] for r in matched]) ** 2, weights=area_w)))
    small = [r for r in matched if r not in big]
    check("inlier_rms_below_1mm", max_rms < TOL_RMS and aw_rms < TOL_RMS,
          {"planes_checked": len(big), "max_rms_to_truth_mm": round(max_rms, 4),
           "area_weighted_rms_all_matched_mm": round(aw_rms, 4),
           "small_planes_max_rms_mm": round(max((r["rms_to_truth_mm"] for r in small), default=0), 4),
           "max_rms_final_to_own_plane_all_mm": float(max(r["rms_final_to_own_plane_mm"] for r in matched)),
           "median_rms_fit_before_snap_mm": round(float(np.median([r["rms_fit_before_snap_mm"] for r in big])), 4),
           "max_rms_final_to_own_plane_mm": float(max(r["rms_final_to_own_plane_mm"] for r in big))})

    # 3. orthogonality in the output frame
    N = {p["id"]: np.array(p["normal"]) for p in rep["planes"]}
    devs = []
    ids = [r["id"] for r in matched]
    for a_ in range(len(ids)):
        for b_ in range(a_ + 1, len(ids)):
            c = abs(N[ids[a_]] @ N[ids[b_]])
            if c < 0.5:
                devs.append(abs(90 - np.degrees(np.arccos(np.clip(c, -1, 1)))))
    check("orthogonal_pairs_90deg", max(devs) < TOL_ANGLE,
          {"pairs": len(devs), "max_dev_from_90_deg": float(max(devs))})

    # 5. metric scale: parallel plane distances vs truth
    sc_err = []
    off = {r["id"]: (np.array(next(p for p in rep["planes"] if p["id"] == r["id"])["normal"]),
                     next(p for p in rep["planes"] if p["id"] == r["id"])["offset"],
                     dict((t[2], t) for t in truth)[r["match"]]) for r in big}
    keys = list(off)
    for a_ in range(len(keys)):
        for b_ in range(a_ + 1, len(keys)):
            na, da, ta = off[keys[a_]]; nb, db, tb = off[keys[b_]]
            if abs(na @ nb) > 0.999:
                d_out = abs(da - (na @ nb) * db)
                d_tru = abs(ta[1] - (ta[0] @ tb[0]) * tb[1])
                sc_err.append(abs(d_out - d_tru))
    check("metric_distances_within_2mm", max(sc_err) < 2.0,
          {"parallel_pairs": len(sc_err), "max_abs_err_mm": round(float(max(sc_err)), 4),
           "mean_abs_err_mm": round(float(np.mean(sc_err)), 4),
           "countertop_height_mm": rep["countertop_height"]})

    # 6. face reduction
    fin, fout = rep["faces_after_floaters"], rep["faces_output"]
    check("face_count_reduced_80pct", fout < 0.2 * fin,
          {"faces_input_raw": rep["faces_input"], "faces_after_floaters": fin,
           "faces_output": fout, "reduction_pct": round(100 * (1 - fout / fin), 2)})

    # 7. topology
    ta = rep["topology_after"]
    check("topology_clean", ta["watertight"] and ta["winding_consistent"]
          and ta["non_manifold_edges"] == 0 and ta["degenerate_faces"] == 0,
          {k: ta[k] for k in ("watertight", "winding_consistent", "non_manifold_edges",
                              "boundary_edges", "degenerate_faces", "duplicate_faces",
                              "connected_components")})

    results["planes"] = rows
    results["layers"] = {k: v["faces"] for k, v in rep["layers"].items()}
    results["decimation"] = rep["decimation"]
    results["manhattan"] = rep["manhattan"]
    results["cad_clean_log_tail"] = proc.stderr.strip().splitlines()[-12:]
    results["all_pass"] = bool(ok_all)
    results["work_dir"] = work
    return results


def test_cad_clean():   # pytest entry point
    r = run()
    assert r["all_pass"], json.dumps(r["checks"], indent=1)


if __name__ == "__main__":
    keep = sys.argv[sys.argv.index("--keep-dir") + 1] if "--keep-dir" in sys.argv else None
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
    r = run(keep, seed)
    print(json.dumps({k: v for k, v in r.items() if k != "planes"}, indent=1))
    print("\nplanes:")
    for row in r["planes"]:
        print("  ", row)
    print("\nALL PASS" if r["all_pass"] else "\nFAILURES")
    sys.exit(0 if r["all_pass"] else 1)
