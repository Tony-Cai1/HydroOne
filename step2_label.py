import os
import torch
import math
from ultralytics import YOLO

# ---- CONFIG ----
MODEL_PATH   = r"./yoloe.pt"               # YOLO model
IMAGES_ROOT  = r"./sweep_out/images"
LABELS_ROOT  = r"./sweep_out/labels"
ROUTE_TAG    = "route_5"
CONF_THR     = 0.25
IOU_THR      = 0.25
BATCH_SIZE   = 16                         # tuned below per device
WORKERS      = max(2, (os.cpu_count() or 4) // 2)
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ---- LOAD MODEL ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
HALF = device == "cuda"

# Adjust defaults for CPU to avoid OOM
if device == "cpu":
    BATCH_SIZE = 4
    WORKERS = 0  # Windows + CPU dataloader stability

def describe_device():
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[INFO] Using GPU: {name} (cc {cap[0]}.{cap[1]}), half={HALF}")
        except Exception:
            print("[INFO] Using GPU")
    else:
        print("[INFO] Using CPU (consider installing a CUDA-enabled PyTorch to use your GPU)")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_labels_for_result(result):
    # origin image info
    img_path = result.path
    h, w = result.orig_shape
    # convert to normalized YOLO lines
    lines = []
    for b in result.boxes:
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        xc = (x1 + x2) / 2.0 / w
        yc = (y1 + y2) / 2.0 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # mirror directory under LABELS_ROOT
    rel_path = os.path.relpath(img_path, IMAGES_ROOT)
    rel_dir = os.path.dirname(rel_path)
    target_dir = os.path.join(LABELS_ROOT, rel_dir) if rel_dir != "." else LABELS_ROOT
    ensure_dir(target_dir)
    txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    txt_path = os.path.join(target_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return txt_path, len(lines)

def collect_image_sources(root: str):
    sources = []
    for cur_dir, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                sources.append(os.path.join(cur_dir, fn))
    return sources

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    ensure_dir(LABELS_ROOT)
    
    # Decide where to search for images
    if ROUTE_TAG:
        route_root = os.path.join(IMAGES_ROOT, ROUTE_TAG)
    else:
        route_root = IMAGES_ROOT

    # Collect all image files under that route only
    sources = collect_image_sources(route_root)
    if not sources:
        raise FileNotFoundError(f"No images found under {route_root}. Ensure images exist in subfolders.")

    describe_device()

    # Process in file-list chunks to keep memory bounded even on CPU
    total_imgs = 0
    total_labels = 0
    FILELIST_CHUNK = 512  # number of files per Ultralytics .predict call
    for src_chunk in chunked(sources, FILELIST_CHUNK):
        results = model.predict(
            source=src_chunk,
            conf=CONF_THR,
            iou=IOU_THR,
            imgsz=640,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=device,
            half=HALF,
            stream=True,
            verbose=False,
        )

        for r in results:
            txt_path, n = write_labels_for_result(r)
            total_imgs += 1
            total_labels += n
            print(f"[ok] {os.path.relpath(r.path)} -> {os.path.relpath(txt_path)} (det={n})")

    print(f"\nDone. Images processed: {total_imgs}, total boxes: {total_labels}")
    print(f"Labels root: {LABELS_ROOT}")

if __name__ == "__main__":
    main()
