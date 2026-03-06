#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redetect (clustered iterative zoom)

For every sweep image that has labels:
- Group sleeves into clusters that fit inside the base FOV (with a small margin).
- For each cluster, center on the cluster centroid and perform 3 zoom steps
  at 1.5× per step (FOV /= 1.5), re-centering slightly from detections to keep
  the cluster in view. Save one final per cluster.

Inputs:
  sweep_out/images[/<route>]/<point>/*.jpg
  sweep_out/labels[/<route>]/<point>/*.txt

Outputs:
  doublecheck_out/images[/<route>]/<point>/dc_<base>_det<i>_final.jpg
  doublecheck_out/labels[/<route>]/<point>/dc_<base>_det<i>_final.txt
  doublecheck_out/visualized[/<route>]/<point>/dc_<base>_det<i>_final.jpg
"""

import os
import re
import math
import time
import glob
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ========================= CONFIG =========================

IMAGES_ROOT  = r"sweep_out/images/route_5"
LABELS_ROOT  = r"sweep_out/labels/route_5"
ROUTE_TAG    = "route_5"

OUT_ROOT    = r"doublecheck_out"
OUT_IMAGES  = os.path.join(OUT_ROOT, "images")
OUT_LABELS  = os.path.join(OUT_ROOT, "labels")
OUT_VIS     = os.path.join(OUT_ROOT, "visualized")

GOOGLE_API_KEY   = "AIzaSyAal2PH1aIbKywxU2JbjxNRIoBrSPBXinY"
REQ_SIZE         = (640, 640)

MODEL_PATH = r"./yoloe.pt"
CONF_THR   = 0.25
IOU_THR    = 0.5

# Iterative cluster zoom config
ZOOM_STEPS        = 3        # number of zoom steps per cluster
ZOOM_FACTOR_STEP  = 1.5      # each step: FOV /= 1.5
CLUSTER_MARGIN_DEG = 5.0     # margin (deg) to keep all sleeves of a cluster in frame
MIN_FOV            = 12.0
SLEEP_SEC          = 0.02

# ==========================================================

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_from_basename(fname: str) -> Dict[str, float]:
    base = os.path.splitext(os.path.basename(fname))[0]
    def grab(tag, cast=float, required=True):
        m = re.search(rf"{tag}(-?\d+(?:\.\d+)?)", base)
        if not m:
            if required:
                raise ValueError(f"Missing {tag} in filename: {fname}")
            return None
        return cast(m.group(1))
    lat = grab("lat")
    mlon = re.search(r"lon(-?\d+(?:\.\d+)?)", base)
    if not mlon:
        raise ValueError(f"Missing lon in filename: {fname}")
    lon = float(mlon.group(1))
    hdg = grab("hdg")
    pit = grab("pitch")
    fov = grab("fov")
    msnap = re.search(r"snap(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", base)
    snap_lat = float(msnap.group(1)) if msnap else None
    snap_lon = float(msnap.group(2)) if msnap else None
    return dict(lat=lat, lon=lon, heading=hdg, pitch=pit, fov=fov,
                snap_lat=snap_lat, snap_lon=snap_lon)

def streetview_metadata(lat: float, lon: float, api_key: str, radius: int = 50):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": radius, "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    return j if j.get("status") == "OK" else None

def streetview_image_pano(pano_id: str, heading: float, pitch: float, fov: float, size=(640,640)):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "pano": pano_id,
        "heading": f"{heading:.6f}",
        "pitch": f"{pitch:.6f}",
        "fov": f"{fov:.6f}",
        "size": f"{size[0]}x{size[1]}",
        "key": GOOGLE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    return img

def fov_vertical(hfov_deg: float, width: int, height: int) -> float:
    hf = math.radians(hfov_deg)
    vf = 2.0 * math.atan((height/width) * math.tan(hf/2.0))
    return math.degrees(vf)

def ang_offsets_from_norm(xc: float, yc: float, hfov_deg: float, vfov_deg: float) -> Tuple[float,float]:
    u = 2.0*xc - 1.0
    v = 1.0 - 2.0*yc
    hf = math.radians(hfov_deg)
    vf = math.radians(vfov_deg)
    dyaw   = math.degrees(math.atan(u * math.tan(hf/2.0)))
    dpitch = math.degrees(math.atan(v * math.tan(vf/2.0)))
    return dyaw, dpitch

def read_labels(lbl_path: str) -> List[Dict[str, float]]:
    out = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc = float(parts[1]); yc = float(parts[2])
                ww = float(parts[3]); hh = float(parts[4])
                out.append({"cls": cls_id, "xc": xc, "yc": yc, "w": ww, "h": hh})
            except Exception:
                continue
    return out

def iter_point_dirs(images_root: str):
    direct_points = [p for p in glob.glob(os.path.join(images_root, "*")) if os.path.isdir(p) and glob.glob(os.path.join(p, "*.jpg"))]
    if direct_points:
        for pdir in sorted(direct_points):
            yield (None, pdir, os.path.basename(pdir))
        return
    for rdir in sorted([d for d in glob.glob(os.path.join(images_root, "*")) if os.path.isdir(d)]):
        route = os.path.basename(rdir)
        for pdir in sorted([p for p in glob.glob(os.path.join(rdir, "*")) if os.path.isdir(p)]):
            yield (route, pdir, os.path.basename(pdir))


def main():
    ensure_dirs(OUT_IMAGES, OUT_LABELS, OUT_VIS)
    model = YOLO(MODEL_PATH)

    any_points = False
    for route, pdir, point_name in iter_point_dirs(IMAGES_ROOT):
        any_points = True

        src_label_dir = os.path.join(LABELS_ROOT, point_name) if route is None else os.path.join(LABELS_ROOT, route, point_name)
        if not os.path.isdir(src_label_dir):
            print(f"[{point_name}] no labels dir; skipping")
            continue

        img_files = sorted(glob.glob(os.path.join(pdir, "*.jpg")))
        if not img_files:
            print(f"[{point_name}] no images; skipping")
            continue

        # labeled images only
        labeled_imgs = []
        for _img in img_files:
            base = os.path.splitext(os.path.basename(_img))[0]
            lbl = os.path.join(src_label_dir, base + ".txt")
            if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                labeled_imgs.append(_img)
        if not labeled_imgs:
            print(f"[{point_name}] no images with labels; skipping point")
            continue

        # pano lock from first labeled image (snapped coords if present)
        try:
            first_info = parse_from_basename(labeled_imgs[0])
        except Exception as e:
            print(f"[{point_name}] cannot parse first labeled image: {e}")
            continue
        m_lat = first_info["snap_lat"] if first_info["snap_lat"] is not None else first_info["lat"]
        m_lon = first_info["snap_lon"] if first_info["snap_lon"] is not None else first_info["lon"]
        meta = streetview_metadata(m_lat, m_lon, GOOGLE_API_KEY, radius=50)
        if not meta or "pano_id" not in meta:
            print(f"[{point_name}] no pano via metadata; skipping point")
            continue
        pano_id = meta["pano_id"]

        for img_path in labeled_imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(src_label_dir, base + ".txt")
            labels = read_labels(lbl_path)
            if not labels:
                continue

            try:
                info = parse_from_basename(img_path)
            except Exception as e:
                print(f"[{point_name}] {img_path}: {e}")
                continue

            # Compute angular offsets for all labels
            vfov0 = fov_vertical(info["fov"], REQ_SIZE[0], REQ_SIZE[1])
            items = []  # {dy, dp, cls}
            for d in labels:
                dy, dp = ang_offsets_from_norm(d["xc"], d["yc"], info["fov"], vfov0)
                items.append({"dy": dy, "dp": dp, "cls": d["cls"]})

            # Cluster sleeves so each cluster fits within base FOV with a margin
            clusters: List[Dict] = []
            for it in items:
                placed = False
                for C in clusters:
                    ymin = min(C["ymin"], it["dy"]); ymax = max(C["ymax"], it["dy"])
                    pmin = min(C["pmin"], it["dp"]); pmax = max(C["pmax"], it["dp"])
                    if (ymax - ymin + 2*CLUSTER_MARGIN_DEG) <= info["fov"] and (pmax - pmin + 2*CLUSTER_MARGIN_DEG) <= vfov0:
                        C["members"].append(it); C["ymin"], C["ymax"], C["pmin"], C["pmax"] = ymin, ymax, pmin, pmax
                        placed = True
                        break
                if not placed:
                    clusters.append({
                        "members": [it],
                        "ymin": it["dy"], "ymax": it["dy"],
                        "pmin": it["dp"], "pmax": it["dp"],
                    })

            for k, C in enumerate(clusters):
                ys = [m["dy"] for m in C["members"]]; ps = [m["dp"] for m in C["members"]]
                cy = sum(ys)/len(ys); cp = sum(ps)/len(ps)
                env_h = (max(ys) - min(ys)) + 2*CLUSTER_MARGIN_DEG
                env_v = (max(ps) - min(ps)) + 2*CLUSTER_MARGIN_DEG
                env   = max(env_h, env_v)

                heading = (info["heading"] + cy) % 360.0
                pitch   = clamp(info["pitch"] + cp, -90.0, 90.0)
                hfov    = info["fov"]

                for step in range(ZOOM_STEPS):
                    img = streetview_image_pano(pano_id, heading, pitch, hfov, REQ_SIZE)
                    if img is None or img.size == 0:
                        print(f"   [cluster{k} step{step}] fetch failed")
                        break
                    r = model.predict(source=img, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
                    preds = []
                    H, W = img.shape[:2]
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                        xc_n = ((x1 + x2) / 2.0) / W
                        yc_n = ((y1 + y2) / 2.0) / H
                        preds.append({"xc": xc_n, "yc": yc_n})
                    if preds:
                        vf = fov_vertical(hfov, W, H)
                        ex_pts = []
                        for m in C["members"]:
                            d_y = m["dy"] - cy
                            d_p = m["dp"] - cp
                            u = math.tan(math.radians(d_y)) / max(1e-6, math.tan(math.radians(hfov/2.0)))
                            v = math.tan(math.radians(d_p)) / max(1e-6, math.tan(math.radians(vf/2.0)))
                            ex = (u + 1.0) * 0.5
                            ey = (1.0 - v) * 0.5
                            nearest = min(preds, key=lambda p: (p["xc"]-ex)**2 + (p["yc"]-ey)**2)
                            ex_pts.append((nearest["xc"], nearest["yc"]))
                        mean_x = sum(p[0] for p in ex_pts)/len(ex_pts)
                        mean_y = sum(p[1] for p in ex_pts)/len(ex_pts)
                        dy_c, dp_c = ang_offsets_from_norm(mean_x, mean_y, hfov, vf)
                        heading = (heading + dy_c) % 360.0
                        pitch   = clamp(pitch + dp_c, -90.0, 90.0)
                    next_h = max(MIN_FOV, hfov / ZOOM_FACTOR_STEP)
                    if next_h < env:
                        hfov = env
                        break
                    hfov = next_h
                    time.sleep(SLEEP_SEC)

                final_img = streetview_image_pano(pano_id, heading, pitch, hfov, REQ_SIZE)
                if final_img is None or final_img.size == 0:
                    print(f"   [cluster{k}] final fetch failed; skipping")
                    continue

                fr = model.predict(source=final_img, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
                fboxes, fclss, fconfs = [], [], []
                Hf, Wf = final_img.shape[:2]
                lines_out = []
                for b in fr.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    fboxes.append([x1, y1, x2, y2])
                    fclss.append(int(b.cls[0]))
                    fconfs.append(float(b.conf[0]))
                    xc = (x1 + x2) / 2.0 / Wf
                    yc = (y1 + y2) / 2.0 / Hf
                    ww = (x2 - x1) / Wf
                    hh = (y2 - y1) / Hf
                    cls_id = int(b.cls[0])
                    conf   = float(b.conf[0])
                    lines_out.append(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f} {conf:.3f}")

                # If no labels after zoom, skip saving (also nothing saved during zoom steps)
                if not fboxes:
                    print(f"   [cluster{k}] no detections after zoom; skipping save")
                    continue

                out_img_dir = os.path.join(OUT_IMAGES, ROUTE_TAG, point_name) if route is None else os.path.join(OUT_IMAGES, route, point_name)
                out_lbl_dir = out_img_dir.replace("/images", "/labels").replace("\\images", "\\labels")
                out_vis_dir = out_img_dir.replace("/images", "/visualized").replace("\\images", "\\visualized")
                ensure_dirs(out_img_dir, out_lbl_dir, out_vis_dir)

                # Update filename tokens to reflect the final view heading/pitch/fov
                base_final = base
                try:
                    base_final = re.sub(r"hdg-?\d+", f"hdg{int(round(heading))%360:03d}", base_final)
                    base_final = re.sub(r"pitch-?\d+", f"pitch{int(round(pitch))}", base_final)
                    base_final = re.sub(r"fov-?\d+", f"fov{int(round(hfov))}", base_final)
                except Exception:
                    pass
                out_base = f"dc_{base_final}_cluster{k}"
                img_out = os.path.join(out_img_dir, out_base + "_final.jpg")
                lbl_out = os.path.join(out_lbl_dir, out_base + "_final.txt")
                vis_out = os.path.join(out_vis_dir, out_base + "_final.jpg")

                cv2.imwrite(img_out, final_img)
                with open(lbl_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines_out))
                vis = final_img.copy()
                for (x1, y1, x2, y2) in fboxes:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.imwrite(vis_out, vis)
                print(f"   [cluster{k}] saved -> {img_out}")
                time.sleep(SLEEP_SEC)

    if not any_points:
        print("No point folders found under", IMAGES_ROOT)
    print("Done. Results under:", OUT_ROOT)

if __name__ == "__main__":
    main()
