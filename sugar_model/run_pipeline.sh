#!/usr/bin/env bash
# run_pipeline.sh - video/photos -> COLMAP -> SuGaR mesh -> cad_clean (metric, Z-up, segmented CAD mesh)
#
# Usage:
#   bash run_pipeline.sh --video kitchen.mov --name smith_kitchen [options]
#   bash run_pipeline.sh --images ./photos  --name smith_kitchen [options]
# Options:
#   --frames N          target frame count from video (default 200; SuGaR README: 100-300)
#   --fps F             override frame rate instead of --frames
#   --reg R             dn_consistency (default, README-recommended) | density | sdf
#   --poly high|low     --high_poly True (1M verts) or --low_poly True (200k verts); default high
#   --refine T          short (2k it) | medium (7k) | long (15k, default in SuGaR); default medium
#   --counter-in H      countertop height in inches for metric scale (default 36)
#   --scale-ref STR     pass-through to cad_clean (e.g. "points=x1,y1,z1;x2,y2,z2;914.4")
#   --units mm|cm|m|in  output units (default mm)
#   --sugar DIR         SuGaR checkout (default ~/SuGaR)
#   --work DIR          job root (default ./jobs)
#   --skip-colmap       reuse existing COLMAP dataset
# Every step logs to <work>/<name>/logs/*.log.
set -euo pipefail

VIDEO=""; IMAGES=""; NAME=""; FRAMES=200; FPS=""; REG="dn_consistency"; POLY="high"
REFINE="medium"; COUNTER_IN="36"; SCALE_REF=""; UNITS="mm"; SUGAR_DIR="$HOME/SuGaR"
WORK="$(pwd)/jobs"; SKIP_COLMAP=0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --video) VIDEO="$(realpath "$2")"; shift 2;;
    --images) IMAGES="$(realpath "$2")"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --frames) FRAMES="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --reg) REG="$2"; shift 2;;
    --poly) POLY="$2"; shift 2;;
    --refine) REFINE="$2"; shift 2;;
    --counter-in) COUNTER_IN="$2"; shift 2;;
    --scale-ref) SCALE_REF="$2"; shift 2;;
    --units) UNITS="$2"; shift 2;;
    --sugar) SUGAR_DIR="$(realpath "$2")"; shift 2;;
    --work) WORK="$(realpath -m "$2")"; shift 2;;
    --skip-colmap) SKIP_COLMAP=1; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
[ -n "$NAME" ] || { echo "--name required"; exit 2; }
[ -n "$VIDEO$IMAGES" ] || [ "$SKIP_COLMAP" = 1 ] || { echo "--video or --images required"; exit 2; }

JOB="$WORK/$NAME"            # COLMAP dataset dir (SuGaR names outputs after this folder)
LOGS="$JOB/logs"; mkdir -p "$JOB/input" "$LOGS"
log() { printf '[pipeline %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOGS/pipeline.log"; }
step() { local n="$1"; shift; log "START $n"; local t0=$SECONDS
         if "$@" >"$LOGS/$n.log" 2>&1; then log "OK    $n ($((SECONDS-t0)) s)";
         else log "FAIL  $n - see $LOGS/$n.log"; tail -30 "$LOGS/$n.log"; exit 1; fi; }

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sugar

# ---------------------------------------------------------------- 1. frames
if [ "$SKIP_COLMAP" = 0 ]; then
  if [ -n "$VIDEO" ]; then
    if [ -z "$FPS" ]; then
      DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
      FPS=$(python -c "print(max(0.2, round($FRAMES/float($DUR), 3)))")
    fi
    log "extracting frames at fps=$FPS (target ~$FRAMES)"
    # exact command from the SuGaR README tips section
    step ffmpeg ffmpeg -y -i "$VIDEO" -qscale:v 1 -qmin 1 -vf fps="$FPS" "$JOB/input/%04d.jpg"
  else
    log "copying photos from $IMAGES"
    find "$IMAGES" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      -exec cp {} "$JOB/input/" \;
  fi
  N=$(ls "$JOB/input" | wc -l); log "$N images"
  [ "$N" -ge 60 ] || log "WARNING: only $N images - expect COLMAP gaps/holes (aim 150-300)"

  # ------------------------------------------------------------ 2. COLMAP
  GPUFLAG=""; colmap -h 2>&1 | grep -qi "with CUDA" || GPUFLAG="--no_gpu"
  cd "$SUGAR_DIR"
  step colmap python gaussian_splatting/convert.py -s "$JOB" $GPUFLAG
  # README: if COLMAP split into several sub-models keep the largest, rename to 0, re-run with --skip_matching
  NSUB=$(find "$JOB/distorted/sparse" -mindepth 1 -maxdepth 1 -type d | wc -l)
  if [ "$NSUB" -gt 1 ]; then
    log "COLMAP produced $NSUB sub-models - keeping the largest (README procedure)"
    BEST=$(for d in "$JOB"/distorted/sparse/*/; do echo "$(stat -c %s "$d/images.bin") $d"; done | sort -nr | head -1 | cut -d' ' -f2)
    mkdir -p "$JOB/distorted/sparse_discarded"
    for d in "$JOB"/distorted/sparse/*/; do [ "$d" = "$BEST" ] || mv "$d" "$JOB/distorted/sparse_discarded/"; done
    [ "$(basename "$BEST")" = "0" ] || mv "$BEST" "$JOB/distorted/sparse/0"
    rm -rf "$JOB/sparse" "$JOB/images"
    step colmap_rerun python gaussian_splatting/convert.py -s "$JOB" --skip_matching $GPUFLAG
  fi
fi
REG_IMG=$(python - "$JOB" <<'EOF'
import struct, sys
with open(sys.argv[1] + "/sparse/0/images.bin", "rb") as f:
    print(struct.unpack("<Q", f.read(8))[0])
EOF
)
log "COLMAP registered $REG_IMG images"

# ---------------------------------------------------------------- 3. SuGaR
cd "$SUGAR_DIR"
POLYFLAG="--high_poly True"; [ "$POLY" = "low" ] && POLYFLAG="--low_poly True"
# flags exactly as documented in the SuGaR README / train_full_pipeline.py
step sugar python train_full_pipeline.py -s "$JOB" -r "$REG" $POLYFLAG \
     --refinement_time "$REFINE" --export_obj True --export_ply True

# SuGaR writes ./output/{coarse_mesh,refined_mesh,refined_ply}/<scene folder name>/
OBJ=$(ls -t "$SUGAR_DIR/output/refined_mesh/$NAME/"*.obj 2>/dev/null | head -1 || true)
PLY=$(ls -t "$SUGAR_DIR/output/coarse_mesh/$NAME/"*.ply 2>/dev/null | head -1 || true)
MESH="${OBJ:-$PLY}"
[ -n "$MESH" ] || { log "no SuGaR mesh found"; exit 1; }
mkdir -p "$JOB/sugar"; cp "$MESH" "$JOB/sugar/"; [ -n "$PLY" ] && cp "$PLY" "$JOB/sugar/" || true
[ -n "$OBJ" ] && cp "$(dirname "$OBJ")"/*.png "$JOB/sugar/" 2>/dev/null || true
log "SuGaR mesh: $MESH"

# ---------------------------------------------------------------- 4. CAD clean
SREF="${SCALE_REF:-counter_height=$(python -c "print($COUNTER_IN*{'mm':25.4,'cm':2.54,'m':0.0254,'in':1}['$UNITS'])")}"
step cad_clean python "$HERE/postprocess/cad_clean.py" "$JOB/sugar/$(basename "$MESH")" "$JOB/cad" \
     --units "$UNITS" --scale-ref "$SREF" --colmap-images "$JOB/sparse/0/images.bin"
python - "$JOB/cad" <<'EOF' | tee -a "$JOB/logs/pipeline.log"
import glob, json, sys
r = json.load(open(glob.glob(sys.argv[1] + "/*_report.json")[0]))
t = r["topology_after"]
print(f"[pipeline] faces {r['faces_input']} -> {r['faces_output']}; planes {len(r['planes'])}; "
      f"countertop {r['countertop_height']}; watertight {t['watertight']}; warnings {r['warnings']}")
EOF
log "DONE -> $JOB/cad  (open *_layers.obj in Blender/Fusion/SketchUp; *_report.json for planes/edges)"
