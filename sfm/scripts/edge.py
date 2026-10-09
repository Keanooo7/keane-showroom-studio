"""edge.py helper: find strongest intensity step along a scan.
edge(img_key, x, y0, y1, 'v') -> y of max |gradient| along column x (avg over +-3 px) between y0..y1
edge(img_key, y, x0, x1, 'h') -> x of max |gradient| along row y.
Prints top-3 candidates with signed gradient.
"""
import cv2, numpy as np, glob, sys
_cache = {}


def load(k):
    if k not in _cache:
        f = glob.glob(f"/home/claude/kitchen/images/{k}*.jpg")[0]
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        _cache[k] = cv2.GaussianBlur(im, (0, 0), 1.5)
    return _cache[k]


def scan(k, c, a0, a1, mode="v", hw=4, top=3, sign=0):
    im = load(k)
    if mode == "v":
        prof = im[a0:a1, c - hw:c + hw + 1].mean(1)
    else:
        prof = im[c - hw:c + hw + 1, a0:a1].mean(0)
    g = np.gradient(prof)
    if sign > 0:
        gg = g.copy(); gg[gg < 0] = 0
    elif sign < 0:
        gg = -g.copy(); gg[gg < 0] = 0
    else:
        gg = np.abs(g)
    # non-max suppression
    idx = [i for i in range(1, len(gg) - 1) if gg[i] >= gg[i - 1] and gg[i] >= gg[i + 1]]
    idx = sorted(idx, key=lambda i: -gg[i])[:top]
    out = []
    for i in idx:
        # subpixel parabola
        if 0 < i < len(gg) - 1:
            d = (gg[i - 1] - gg[i + 1]) / (2 * (gg[i - 1] - 2 * gg[i] + gg[i + 1]) + 1e-9)
        else:
            d = 0
        out.append((round(a0 + i + d, 1), round(float(g[i]), 1)))
    return out


if __name__ == "__main__":
    k, c, a0, a1, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    print(scan(k, c, a0, a1, mode))
