"""Agent A: SfM with pycolmap on downscaled images.

Usage: python run_sfm.py <variant>
Variants control feature/matching/mapping settings.
Work dir: /home/claude/kitchen/artifacts/sfm/work_<variant>
"""
import sys, os, shutil, time, json
import cv2
import numpy as np
import pycolmap

ROOT = "/home/claude/kitchen/artifacts/sfm"
SRC = "/home/claude/kitchen/images"
NAMES = ["01_sink_closeup", "02_corner_wide", "03_corner_high", "04_right_run", "05_left_run"]
variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
SCALE = float(os.environ.get("SFM_SCALE", "0.5"))  # downscale factor (0.5 -> 1512x2016)
W, H = int(round(3024 * SCALE)), int(round(4032 * SCALE))
F0 = 3028.0 * SCALE

work = f"{ROOT}/work_{variant}"
imgdir = f"{work}/images"
if os.path.exists(work):
    shutil.rmtree(work)
os.makedirs(f"{imgdir}/cam1", exist_ok=True)
os.makedirs(f"{imgdir}/cam2", exist_ok=True)
for n in NAMES:
    im = cv2.imread(f"{SRC}/{n}.jpg")
    s = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    sub = "cam1" if n.startswith("01") else "cam2"
    cv2.imwrite(f"{imgdir}/{sub}/{n}.jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 97])

db = f"{work}/database.db"
t0 = time.time()
reader = pycolmap.ImageReaderOptions()
reader.camera_model = os.environ.get("SFM_CAMMODEL", "SIMPLE_RADIAL")
if reader.camera_model == "SIMPLE_RADIAL":
    reader.camera_params = f"{F0},{W/2},{H/2},0"
elif reader.camera_model == "OPENCV":
    reader.camera_params = f"{F0},{F0},{W/2},{H/2},0,0,0,0"
ext = pycolmap.FeatureExtractionOptions()
ext.use_gpu = False
ext.num_threads = 2
ext.max_image_size = max(W, H)
ext.sift.max_num_features = int(os.environ.get("SFM_MAXFEAT", "16384"))
ext.sift.estimate_affine_shape = os.environ.get("SFM_AFFINE", "1") == "1"
ext.sift.domain_size_pooling = os.environ.get("SFM_DSP", "1") == "1"
ext.sift.peak_threshold = float(os.environ.get("SFM_PEAK", "0.004"))
ext.sift.first_octave = -1
pycolmap.extract_features(db, imgdir, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                          reader_options=reader, extraction_options=ext,
                          device=pycolmap.Device.cpu)
print(f"extract: {time.time()-t0:.1f}s", flush=True)

# mark prior focal length
dbo = pycolmap.Database.open(db)
for cam in dbo.read_all_cameras():
    cam.has_prior_focal_length = True
    dbo.update_camera(cam)
for im in dbo.read_all_images():
    print(im.image_id, im.name, im.camera_id, dbo.num_keypoints_for_image(im.image_id))
dbo.close()

t0 = time.time()
mo = pycolmap.FeatureMatchingOptions()
mo.use_gpu = False
mo.num_threads = int(os.environ.get("SFM_MTHREADS", "2"))
mo.guided_matching = os.environ.get("SFM_GUIDED", "1") == "1"
mo.sift.max_ratio = float(os.environ.get("SFM_RATIO", "0.85"))
mo.sift.max_distance = 0.7
mo.sift.cross_check = True
vo = pycolmap.TwoViewGeometryOptions()
vo.min_num_inliers = int(os.environ.get("SFM_MININL", "15"))
vo.ransac.max_error = float(os.environ.get("SFM_RANSACERR", "4.0"))
vo.ransac.max_num_trials = 50000
vo.ransac.confidence = 0.9999
vo.detect_watermark = False
pycolmap.match_exhaustive(db, matching_options=mo, verification_options=vo, device=pycolmap.Device.cpu)
print(f"match: {time.time()-t0:.1f}s", flush=True)

dbo = pycolmap.Database.open(db)
id2name = {im.image_id: im.name for im in dbo.read_all_images()}
pairs = {}
for (pid, tvg) in zip(*dbo.read_two_view_geometries()) if False else []:
    pass
try:
    pids, tvgs = dbo.read_two_view_geometries()
    for pid, tvg in zip(pids, tvgs):
        a, b = pycolmap.pair_id_to_image_pair(pid)
        pairs[f"{id2name[a]}|{id2name[b]}"] = (int(tvg.inlier_matches.shape[0]), str(tvg.config))
except Exception as e:
    print("tvg read err", e)
try:
    allm = dbo.read_all_matches()
    raw = {}
    for pid, m in zip(*allm):
        a, b = pycolmap.pair_id_to_image_pair(pid)
        raw[f"{id2name[a]}|{id2name[b]}"] = int(m.shape[0])
except Exception as e:
    raw = {}
    print("matches read err", e)
dbo.close()
for k in sorted(set(list(pairs) + list(raw))):
    print(f"  {k}: raw={raw.get(k)} verified={pairs.get(k)}")
json.dump({"raw": raw, "verified": pairs}, open(f"{work}/pair_stats.json", "w"), indent=1)

t0 = time.time()
po = pycolmap.IncrementalPipelineOptions()
po.min_num_matches = int(os.environ.get("SFM_MINMATCH", "15"))
po.num_threads = 2
po.multiple_models = True
po.min_model_size = 2
po.ba_refine_focal_length = True
po.ba_refine_extra_params = True
po.mapper.init_min_num_inliers = int(os.environ.get("SFM_INITINL", "50"))
po.mapper.init_min_tri_angle = float(os.environ.get("SFM_INITANG", "8"))
po.mapper.abs_pose_min_num_inliers = int(os.environ.get("SFM_ABSINL", "15"))
po.mapper.abs_pose_min_inlier_ratio = 0.1
po.mapper.max_reg_trials = 5
po.mapper.init_max_reg_trials = 4
po.triangulation.ignore_two_view_tracks = False
os.makedirs(f"{work}/sparse", exist_ok=True)
recs = pycolmap.incremental_mapping(db, imgdir, f"{work}/sparse", options=po)
print(f"map: {time.time()-t0:.1f}s", flush=True)
for k, r in recs.items():
    print("model", k, r.summary())
    print("  registered:", sorted(im.name for im in r.images.values() if im.has_pose))
    for c in r.cameras.values():
        print("  cam", c.camera_id, c.model, c.params)
