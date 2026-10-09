"""crop.py <img_prefix> <cx> <cy> <half> [out_size] [tag]
Crops a square around (cx,cy) in full-res coords, upsamples to out_size, draws a tick grid
labeled in full-res pixel coords."""
import sys, cv2, glob
p, cx, cy, half = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out = int(sys.argv[5]) if len(sys.argv) > 5 else 800
tag = sys.argv[6] if len(sys.argv) > 6 else ""
f = glob.glob(f"/home/claude/kitchen/images/{p}*.jpg")[0]
im = cv2.imread(f)
x0, y0 = max(cx - half, 0), max(cy - half, 0)
x1, y1 = min(cx + half, im.shape[1]), min(cy + half, im.shape[0])
c = im[y0:y1, x0:x1].copy()
s = out / max(c.shape[:2])
c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
# grid
step = 10
for st in [10, 20, 25, 50, 100, 200, 250, 500]:
    if (2 * half) / st <= 16:
        step = st
        break
import math
gx = math.ceil(x0 / step) * step
while gx < x1:
    X = int((gx - x0) * s)
    cv2.line(c, (X, 0), (X, c.shape[0]), (0, 0, 255), 1)
    cv2.putText(c, str(gx), (X + 2, 12), 0, 0.4, (0, 0, 255), 1)
    gx += step
gy = math.ceil(y0 / step) * step
while gy < y1:
    Y = int((gy - y0) * s)
    cv2.line(c, (0, Y), (c.shape[1], Y), (255, 0, 0), 1)
    cv2.putText(c, str(gy), (2, Y - 2), 0, 0.4, (255, 0, 0), 1)
    gy += step
o = f"/home/claude/kitchen/artifacts/sfm/scripts/view/crop_{p}_{cx}_{cy}_{half}{tag}.jpg"
cv2.imwrite(o, c)
print(o)
