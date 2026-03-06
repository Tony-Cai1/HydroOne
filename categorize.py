import os
import csv
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH   = r"./yoloe.pt"
IMAGES_ROOT  = r"./sweep_out/images"
LABELS_ROOT  = r"./sweep_out/labels"

ROUTE_TAG    = "route_5"

IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

IMG_SIZE      = 640
IOU_THR       = 0.25
PRED_MIN_CONF = 0.01

BOTTOMLINE     = 0.15
REVIEW_LOW     = 0.35
CONCLUSIVE_LOW = 0.70

WRITE_CONF_THR = REVIEW_LOW

BATCH_SIZE   = 16
WORKERS      = max(2, (os.cpu_count() or 4) // 2)

CATEGORIZED_ROOT = r"./sweep_out/categorized_images"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
HALF = device == "cuda"

if device == "cpu":
    BATCH_SIZE = 4
    WORKERS = 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_image_sources(root):
    sources = []
    for cur_dir, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                sources.append(os.path.join(cur_dir, fn))
    return sources


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def category_from_max_conf(max_conf):
    if max_conf >= CONCLUSIVE_LOW:
        return "conclusive"
    if max_conf >= REVIEW_LOW:
        return "further_review"
    if (max_conf > BOTTOMLINE) and (max_conf < REVIEW_LOW):
        return "inconclusive"
    return None


def write_labels(img_path, orig_shape_hw, boxes):
    h, w = orig_shape_hw
    lines = []

    for b in boxes:
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    rel_path = os.path.relpath(img_path, IMAGES_ROOT)
    target_dir = os.path.join(LABELS_ROOT, os.path.dirname(rel_path))
    ensure_dir(target_dir)

    txt_path = os.path.join(
        target_dir,
        os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path, len(lines)


def load_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(img_path, out_path, category, boxes):
    ensure_dir(os.path.dirname(out_path))

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font()

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])

        label = f"{category} cls={cls_id} conf={conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], width=3)

        tx = max(0, int(x1))
        ty = max(0, int(y1) - 18)

        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb)
        draw.text((tx, ty), label, font=font)

    img.save(out_path, quality=95)


def categorized_image_path(img_path, category):
    rel = os.path.relpath(img_path, IMAGES_ROOT)
    return os.path.join(CATEGORIZED_ROOT, category, rel)


def main():
    ensure_dir(LABELS_ROOT)
    ensure_dir(CATEGORIZED_ROOT)

    root = os.path.join(IMAGES_ROOT, ROUTE_TAG) if ROUTE_TAG else IMAGES_ROOT
    sources = collect_image_sources(root)

    summary_csv = os.path.join(CATEGORIZED_ROOT, "summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "image",
            "category",
            "max_conf",
            "labels_written",
            "annotated_image"
        ])

        for chunk in chunked(sources, 512):
            results = model.predict(
                source=chunk,
                conf=PRED_MIN_CONF,
                iou=IOU_THR,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                workers=WORKERS,
                device=device,
                half=HALF,
                stream=True,
                verbose=False
            )

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                max_conf = float(r.boxes.conf.max())
                category = category_from_max_conf(max_conf)

                if category is None:
                    continue

                label_boxes = []
                draw_boxes_list = []

                for b in r.boxes:
                    c = float(b.conf[0])

                    if category in ("conclusive", "further_review"):
                        if c >= WRITE_CONF_THR:
                            label_boxes.append(b)
                            draw_boxes_list.append(b)

                    else:
                        if (c > BOTTOMLINE) and (c < REVIEW_LOW):
                            draw_boxes_list.append(b)

                txt_path, n_written = write_labels(r.path, r.orig_shape, label_boxes)

                out_img = categorized_image_path(r.path, category)
                draw_boxes(r.path, out_img, category, draw_boxes_list)

                writer.writerow([
                    os.path.relpath(r.path, IMAGES_ROOT),
                    category,
                    f"{max_conf:.4f}",
                    n_written,
                    out_img
                ])

                print(f"[ok] {r.path} -> {category} drawn={len(draw_boxes_list)}")

    print("Done")


if __name__ == "__main__":
    main()
