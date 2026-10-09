import sys, json, numpy as np
sys.path.insert(0, "/home/claude/kitchen/artifacts/sfm/scripts")
from manhattan import MFrame
p = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/kitchen/artifacts/sfm/work_s075/sparse/0"
R = np.load(p + "/../R_align_vp.npy")
F = MFrame(p, R)
E = {
 "E1_left_front_vertical": (2, {"05": [(1546, 2700), (1546, 2850)], "02": [(348, 2900), (358, 3150)], "03": [(990, 3250), (986, 3400)]}),
 "E2_right_vertical": (2, {"05": [(2401, 2650), (2396, 2780)], "02": [(1368, 2600), (1369, 2800)], "04": [(882, 3150), (888, 3400)]}),
 "E3_top_front_lip": (0, {"05": [(1700, 2625), (2200, 2590)], "02": [(600, 2750), (1200, 2550)], "03": [(1100, 3088), (1400, 2786)]}),
 "E4_bottom_edge": (0, {"05": [(1700, 2928), (2200, 2880)], "02": [(600, 3183), (1300, 2869)], "03": [(1100, 3374), (1400, 3030)]}),
}
res = {}
for nm, (ax, obs) in E.items():
    r = F.line(ax, obs)
    res[nm] = r
    print(nm, r["free"], np.round(r["p"], 4), "sig", np.round(r["sig"], 4), "rms px", round(r["rms"], 2), "max", round(r["maxres"], 2))
    # leave-one-view-out
    for k in obs:
        sub = {kk: v for kk, v in obs.items() if kk != k}
        if len(sub) >= 2:
            r2 = F.line(ax, sub)
            print("     w/o", k, np.round(r2["p"], 4), "rms", round(r2["rms"], 2))
w = res["E1_left_front_vertical"]["p"][0] - res["E2_right_vertical"]["p"][0]
h = res["E3_top_front_lip"]["p"][1] - res["E4_bottom_edge"]["p"][1]
print("width units", w, "height units", h, "ratio h/w", h / w, "expected", 9.875 / 30)
print("apron plane y estimates: E1", res["E1_left_front_vertical"]["p"][1], "E2", res["E2_right_vertical"]["p"][1],
      "E3", res["E3_top_front_lip"]["p"][0], "E4", res["E4_bottom_edge"]["p"][0])
print("scale in/unit from width", 30 / w, "from height", 9.875 / h)
json.dump({k: dict(p=v["p"].tolist(), sig=v["sig"].tolist(), rms=v["rms"], free=v["free"]) for k, v in res.items()},
          open(p + "/../sink_edges.json", "w"), indent=1)
