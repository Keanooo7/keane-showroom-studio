"""Refine approximate picks on vertical edges with horizontal gradient scans, then fit
Manhattan vertical lines (x,y) and report. Writes refined picks + fits to JSON."""
import sys, json, numpy as np
sys.path.insert(0, "/home/claude/kitchen/artifacts/sfm/scripts")
from edge import scan
from manhattan import MFrame

P = "/home/claude/kitchen/artifacts/sfm/work_s075/sparse/0"
F = MFrame(P, np.load(P + "/../R_align_vp.npy"))

# name: {view: [(x_approx, y), ...]}   (two y-levels per view)
V = {
 # left-run uppers
 "U_tallL_left":   {"05": [(925, 1100), (928, 1700)], "03": [(168, 1000), (190, 1500)]},
 "U_short_left":   {"05": [(1432, 900), (1432, 1500)], "03": [(432, 900), (445, 1400)]},
 "U_short_right":  {"05": [(2226, 1000), (2226, 1500)], "02": [(993, 800), (995, 1200)], "03": [(1168, 1000), (1175, 1500)]},
 "U_tallR_right":  {"05": [(2745, 1000), (2745, 1600)], "02": [(1410, 800), (1412, 1300)], "03": [(1440, 1000), (1448, 1500)]},
 # left-run bases
 "B_dod_right":    {"05": [(1515, 2700), (1515, 2850)], "02": [(270, 2900), (288, 3150)], "03": [(914, 3250), (912, 3400)]},
 "B_sink_mid":     {"05": [(1960, 3150), (1960, 3400)], "02": [(896, 3300), (898, 3550)]},
 "B_3dr_left":     {"05": [(2368, 3150), (2368, 3400)], "02": [(1325, 3000), (1327, 3400)], "03": [(1424, 3050), (1430, 3300)]},
 # right-run bases
 "R_door_left":    {"04": [(1122, 3200), (1126, 3700)], "03": [(1690, 2800), (1695, 3100)], "02": [(1780, 2800), (1782, 3200)]},
 "R_seam":         {"04": [(1945, 3200), (1945, 3600)], "03": [(2280, 2800), (2285, 3200)], "02": [(2460, 2800), (2462, 3200)]},
 "R_stack_right":  {"04": [(2752, 3000), (2752, 3600)], "03": [(2940, 2800), (2945, 3300)]},
 # right-run uppers
 "R_up_left":      {"04": [(600, 900), (602, 1700)], "03": [(1520, 1200), (1530, 1700)], "02": [(1640, 900), (1640, 1500)]},
 "R_up_seam1":     {"04": [(1000, 900), (1002, 1700)], "03": [(1760, 1200), (1770, 1700)], "02": [(1900, 900), (1900, 1500)]},
 "R_up_seam2":     {"04": [(1810, 900), (1810, 1700)], "03": [(2350, 1200), (2360, 1700)], "02": [(2580, 900), (2580, 1500)]},
 "R_up_right":     {"04": [(2652, 800), (2652, 1600)], "03": [(3000, 1200), (3005, 1600)]},
}
if __name__ == "__main__":
    win = 30
    out = {}
    for nm, obs in V.items():
        ref = {}
        print(nm)
        for k, pts in obs.items():
            rp = []
            for (x, y) in pts:
                c = scan(k, y, int(x - win), int(x + win), "h", top=3)
                xr = c[0][0]
                rp.append((float(xr), float(y)))
                print(f"   {k} approx {x} -> {xr}  cands {c}")
            ref[k] = rp
        r = F.line(2, ref)
        print(f"   FIT x={r['p'][0]:.4f} y={r['p'][1]:.4f} sig={np.round(r['sig'],4)} rms={r['rms']:.2f} max={r['maxres']:.2f}")
        out[nm] = dict(obs=ref, p=r["p"].tolist(), sig=r["sig"].tolist(), rms=r["rms"], maxres=r["maxres"])
    json.dump(out, open("/home/claude/kitchen/artifacts/sfm/work_s075/vert_edges.json", "w"), indent=1)
