"""Export sparse models (native + aligned inches), PLY, cameras.json, stats.json, coverage.json."""
import json, os, shutil, numpy as np, pycolmap

ROOT = "/home/claude/kitchen/artifacts/sfm"
P = f"{ROOT}/work_s075/sparse/0"
R = np.load(f"{ROOT}/work_s075/sparse/R_align_vp.npy")
meas = json.load(open(f"{ROOT}/scripts/measurements_raw.json"))
s = meas["frame"]["aligned_to_frame"]["inches_per_unit"]
o = np.array(meas["frame"]["aligned_to_frame"]["origin_aligned_units"])
FULL_W, FULL_H = 3024, 4032

rec = pycolmap.Reconstruction(P)
# ---------- native model ----------
for sub, writer in [("sparse_native/bin", "write_binary"), ("sparse_native/txt", "write_text")]:
    d = f"{ROOT}/{sub}"; os.makedirs(d, exist_ok=True)
    getattr(rec, writer)(d)
# ---------- aligned model (inches) ----------
rec_a = pycolmap.Reconstruction(P)
sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(R), -s * o)
rec_a.transform(sim)
for sub, writer in [("sparse_aligned_inches/bin", "write_binary"), ("sparse_aligned_inches/txt", "write_text")]:
    d = f"{ROOT}/{sub}"; os.makedirs(d, exist_ok=True)
    getattr(rec_a, writer)(d)

# ---------- PLY (aligned inches, colored) ----------
ids = sorted(rec_a.points3D.keys())
X = np.array([rec_a.points3D[i].xyz for i in ids])
C = np.array([rec_a.points3D[i].color for i in ids]).astype(np.uint8)
E = np.array([rec_a.points3D[i].error for i in ids])
TL = np.array([rec_a.points3D[i].track.length() for i in ids])
with open(f"{ROOT}/sparse_points.ply", "w") as f:
    f.write("ply\nformat ascii 1.0\ncomment units=inches frame: z up, W1 (sink wall backsplash face) y=0, W2 (right wall backsplash face) x=0, origin inside corner at floor\n")
    f.write(f"element vertex {len(X)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float reproj_error_px\nproperty uchar track_len\nend_header\n")
    for p, c, e, t in zip(X, C, E, TL):
        f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]} {e:.3f} {t}\n")
# also a native-frame PLY via pycolmap
rec.export_PLY(f"{ROOT}/sparse_native/sparse_points_native.ply")

# ---------- cameras.json ----------
cams = {"frame": meas["frame"]["description"], "units": "inches (translation, centers); pixels at FULL resolution 3024x4032",
        "model_note": "SIMPLE_RADIAL: u = f*(x*(1+k*r2)) + cx, with x,y = Xc/Zc, r2=x^2+y^2; pixel coords continuous, (0,0)=top-left corner of top-left pixel (COLMAP convention). SfM was run on 0.75-scale images (2268x3024); intrinsics below are rescaled to full resolution.",
        "images": {}}
for im in sorted(rec_a.images.values(), key=lambda i: i.name):
    cam = rec_a.cameras[im.camera_id]
    sc = FULL_W / cam.width
    f_, cx, cy, k = cam.params
    K = [[f_ * sc, 0, cx * sc], [0, f_ * sc, cy * sc], [0, 0, 1]]
    T = im.cam_from_world().matrix()
    Rw, t = T[:, :3], T[:, 3]
    C_ = -Rw.T @ t
    c2w = np.eye(4); c2w[:3, :3] = Rw.T; c2w[:3, 3] = C_
    cams["images"][im.name.split("/")[-1]] = dict(
        camera_id=int(im.camera_id), camera_model="SIMPLE_RADIAL", width=FULL_W, height=FULL_H,
        K=K, distortion=dict(k1=float(k)), f_px=f_ * sc,
        R_world_to_cam=Rw.tolist(), t_world_to_cam=t.tolist(), cam_to_world_4x4=c2w.tolist(),
        camera_center=C_.tolist(), viewing_direction_world=Rw[2].tolist(),
        num_observed_points3D=int(im.num_points3D), num_keypoints=int(len(im.points2D)))
json.dump(cams, open(f"{ROOT}/cameras.json", "w"), indent=1)

# ---------- stats ----------
tl_hist = {int(k): int(v) for k, v in zip(*np.unique(TL, return_counts=True))}
pair = json.load(open(f"{ROOT}/work_s075/pair_stats.json"))
stats = dict(model="work_s075 (SIFT on 0.75-scale images, 10k max features, DSP+affine SIFT, exhaustive matching, guided matching, incremental mapper)",
             registered_images=sorted(i.name.split('/')[-1] for i in rec.images.values() if i.has_pose),
             num_images_input=5, num_points3D=int(len(X)), num_observations=int(TL.sum()),
             mean_reprojection_error_px_at_0p75=float(rec.compute_mean_reprojection_error()),
             mean_reprojection_error_px_fullres=float(rec.compute_mean_reprojection_error() / 0.75),
             median_point_error_px_at_0p75=float(np.median(E)),
             mean_track_length=float(TL.mean()), track_length_histogram=tl_hist,
             points_per_image={i.name.split('/')[-1]: int(i.num_points3D) for i in rec.images.values()},
             cameras={str(c.camera_id): dict(model=c.model_name, params_at_0p75=list(map(float, c.params)), f_fullres=float(c.params[0] / 0.75)) for c in rec.cameras.values()},
             two_view_verified_inliers=pair["verified"])
json.dump(stats, open(f"{ROOT}/stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k not in ("two_view_verified_inliers",)}, indent=1))

# ---------- coverage ----------
V = {k: v["in_"] for k, v in meas["values"].items()}
zc = V["countertop_top_height_above_floor"]
yuf = V["left_run_upper_door_face_y (depth from W1 backsplash)"]; xuf = V["right_run_upper_door_face_x (depth from W2 backsplash)"]
ybf = V["left_run_base_door_face_y (depth from W1 backsplash)"]; xbf = V["right_run_base_door_face_x (depth from W2 backsplash)"]
zub = V["right_run_upper_bottom_height"]; zcr = V["left_run_crown_underside_height (top of upper boxes)"]
yct = V["left_run_countertop_front_edge_y"]; xct = V["right_run_countertop_front_edge_x"]
yend = V["right_run_countertop_end_y (run length from W2... along W1 normal)"]
xp_r = V["left_run_pantry_right_edge_x"]; xp_l = V["left_run_pantry_left_edge_x"]
ysink = V["sink_apron_front_plane_y"]; xs_l = V["left_run_sink_apron_left_outer_x"]; xs_r = V["left_run_sink_apron_right_outer_x"]
zsb = V["sink_apron_bottom_height"]; zst = V["sink_apron_top_height"]; ztk = V["left_run_base_door_bottom_height (toe-kick top)"]
x, y, z = X[:, 0], X[:, 1], X[:, 2]
B = lambda c: int(c.sum())
regions = {
 "W1 backsplash (counter to upper bottoms, x 0..pantry)": B((y > -1.5) & (y < 2.5) & (z > zc + 0.5) & (z < zub) & (x > 0) & (x < xp_r)),
 "W2 backsplash (counter to upper bottoms)": B((x > -1.5) & (x < 2.5) & (z > zc + 0.5) & (z < zub) & (y > 0) & (y < yend)),
 "countertop top surface (+-0.6in)": B(np.abs(z - zc) < 0.6),
 "countertop front edge faces (left+right, 1.6in band)": B(((np.abs(y - yct) < 0.8) & (x > xbf) & (x < xp_r) | (np.abs(x - xct) < 0.8) & (y > ybf) & (y < yend)) & (z > zc - 1.8) & (z < zc + 0.2)),
 "countertop right end face": B((np.abs(y - yend) < 1.0) & (x < xct + 1) & (z > zc - 2) & (z < zc + 0.3)),
 "left-run base door/drawer faces": B((np.abs(y - ybf) < 0.75) & (z > ztk) & (z < zc - 1.2) & (x > xbf) & (x < xp_r)),
 "right-run base door/drawer faces": B((np.abs(x - xbf) < 0.75) & (z > ztk) & (z < zc - 1.2) & (y > ybf) & (y < yend)),
 "left-run upper door faces": B((np.abs(y - yuf) < 0.75) & (z > zub) & (z < zcr) & (x > xuf) & (x < xp_r)),
 "right-run upper door faces": B((np.abs(x - xuf) < 0.75) & (z > zub) & (z < zcr) & (y > yuf) & (y < yend)),
 "crown molding faces (z crown underside..+4in)": B((z > zcr - 0.5) & (z < zcr + 4.5) & (((np.abs(y - yuf - 1) < 1.5) & (x < xp_r)) | ((np.abs(x - xuf - 1) < 1.5) & (y < yend)))),
 "tops of upper cabinets / crown tops (z>crown top, inside footprint)": B((z > zcr + 3.0) & (z < zcr + 10) & (((y > 1) & (y < yuf - 0.5) & (x > 1) & (x < xp_r)) | ((x > 1) & (x < xuf - 0.5) & (y > 1) & (y < yend)))),
 "undersides of uppers (+-0.75in of bottom, inside footprint)": B((np.abs(z - zub) < 0.75) & (((y > 0.5) & (y < yuf - 0.5) & (x < xp_r)) | ((x > 0.5) & (x < xuf - 0.5) & (y < yend)))),
 "sink apron face": B((np.abs(y - ysink) < 0.6) & (x > xs_r) & (x < xs_l) & (z > zsb) & (z < zst)),
 "sink bowl interior": B((x > xs_r + 1) & (x < xs_l - 1) & (y > ysink - 17) & (y < ysink - 1) & (z > zsb) & (z < zst - 0.3)),
 "toe kicks (z<toe-kick top, 1..4in behind door faces)": B((z > 0.3) & (z < ztk) & ((((ybf - y) > 1) & ((ybf - y) < 4.5) & (x > xbf) & (x < xp_r)) | (((xbf - x) > 1) & ((xbf - x) < 4.5) & (y > ybf) & (y < yend)))),
 "floor (|z|<0.75)": B(np.abs(z) < 0.75),
 "pantry front face": B((np.abs(y - V["pantry_face_y (depth from W1 backsplash)"]) < 0.75) & (x > xp_r) & (x < xp_l) & (z > ztk) & (z < 96)),
 "pantry right side panel (x~pantry right, y 0..24)": B((np.abs(x - xp_r) < 0.75) & (y > 1) & (y < 23) & (z > 0) & (z < 96)),
 "upper side panel visible under short upper (x~42.3, z 55..67)": B((np.abs(x - V["left_run_upper_short_right = tall_right_left_x"]) < 1.0) & (y > 0.5) & (y < yuf - 0.5) & (z > zub) & (z < V["left_run_short_upper_over_sink_bottom_height"])),
 "wall above crown (z>crown top)": B((z > V["left_run_crown_top_height"]) & (((y > -2) & (y < 2)) | ((x > -2) & (x < 2)))),
 "outliers / far points (outside 0..130in box)": B((x < -5) | (y < -5) | (z < -5) | (x > 130) | (y > 130) | (z > 130)),
}
cov = dict(total_points=int(len(X)), regions=regions)
json.dump(cov, open(f"{ROOT}/coverage.json", "w"), indent=1)
for k, v in regions.items():
    print(f"{v:6d}  {k}")
