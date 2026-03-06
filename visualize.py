import os
import cv2
import numpy as np

# ========= CONFIG =========
IMAGE_ROOT   = r"./sweep_out/images"      # traverse images
LABEL_ROOT   = r"./sweep_out/labels"      # mirrored folder structure
VISUAL_ROOT  = r"./sweep_out/visualized"  # images with boxes
DETECT_ROOT  = r"./sweep_out/detected"    # original images, saved only if det>0
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ========= HELPERS =========
def has_image_ext(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS

# Robust Unicode safe read write for Windows paths
def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, image) -> bool:
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path = path + ext
    ok, enc = cv2.imencode(ext, image)
    if not ok:
        return False
    enc.tofile(path)
    return True

def load_yolo_labels(txt_path: str):
    """
    Reads YOLO txt file lines: cls cx cy w h
    Returns list of tuples (cls:int, cx:float, cy:float, w:float, h:float) in normalized units
    Ignores malformed lines gracefully
    """
    items = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                # basic sanity checks
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
                items.append((cls, cx, cy, w, h))
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return items

def draw_boxes(img, labels_norm):
    """
    Draws boxes onto a copy of img using normalized YOLO labels
    Returns the drawn image
    """
    H, W = img.shape[:2]
    vis = img.copy()
    for cls, cx, cy, bw, bh in labels_norm:
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        # clamp
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(cls), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis

# ========= PREP ROOTS =========
os.makedirs(VISUAL_ROOT, exist_ok=True)
os.makedirs(DETECT_ROOT, exist_ok=True)

# ========= WALK AND RENDER =========
total_imgs = 0
total_boxes = 0
total_with_det = 0

for cur_dir, _, files in os.walk(IMAGE_ROOT):
    rel_dir = os.path.relpath(cur_dir, IMAGE_ROOT)
    if rel_dir == ".":
        rel_dir = ""

    # map to mirrored directories
    label_dir = os.path.join(LABEL_ROOT, rel_dir)
    visual_dir = os.path.join(VISUAL_ROOT, rel_dir) if rel_dir else VISUAL_ROOT
    detect_dir = os.path.join(DETECT_ROOT, rel_dir) if rel_dir else DETECT_ROOT

    for fname in files:
        if not has_image_ext(fname):
            continue

        total_imgs += 1
        img_path = os.path.join(cur_dir, fname)
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(label_dir, base + ".txt")

        labels_norm = load_yolo_labels(txt_path)
        if not labels_norm:
            print(f"[ok] {os.path.relpath(img_path)} (det=0, skipped)")
            continue

        img = imread_unicode(img_path)
        if img is None:
            print(f"[warn] Could not read image: {img_path}")
            continue

        # draw boxes to visualized
        os.makedirs(visual_dir, exist_ok=True)
        vis_img = draw_boxes(img, labels_norm)
        out_vis_path = os.path.join(visual_dir, fname)
        if not imwrite_unicode(out_vis_path, vis_img):
            print(f"[warn] Could not write visualization: {out_vis_path}")

        # save original to detected
        os.makedirs(detect_dir, exist_ok=True)
        out_det_path = os.path.join(detect_dir, fname)
        if not imwrite_unicode(out_det_path, img):
            print(f"[warn] Could not write detected original: {out_det_path}")

        total_with_det += 1
        total_boxes += len(labels_norm)
        print(f"[ok] {os.path.relpath(img_path)} -> {os.path.relpath(txt_path)} "
              f"(det={len(labels_norm)}) -> vis:{os.path.relpath(out_vis_path)} -> det:{os.path.relpath(out_det_path)}")

print(f"\nDone. Images scanned: {total_imgs}, images with detections: {total_with_det}, total boxes: {total_boxes}")
print(f"Visualized root: {VISUAL_ROOT}")
print(f"Detected root: {DETECT_ROOT}")
