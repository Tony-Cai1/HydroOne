#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sleeve_counter.py — doublecheck_out version (no API)
----------------------------------------------------
Reads double-checked zoom images/labels, converts detections to bearings,
triangulates between different camera points to estimate sleeve coordinates,
clusters sleeves into sites, and writes:

  sleeves.csv   (lat,lon,number)

Input tree (only detections with final zoom saved will exist):
  doublecheck_out/images/<POINT>/*.jpg
  doublecheck_out/labels/<POINT>/*.txt

Expected filename pattern (inherited from the original base name):
  dc_sweep_lat{lat}_lon{lon}_snap{SLAT}_{SLON}_pano{...}_hdg{H}_pitch{P}_fov{F}_{W}x{H}_final.jpg
The parser searches for tokens (e.g., snap lat/lon if present), so extra prefixes/suffixes are OK.

Label format per line:
  <cls> <xc> <yc> <w> <h> [conf]   # normalized (0..1); conf optional
"""

import os
import re
import csv
import glob
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

# ====================== CONFIG ======================

IMAGES_ROOT = r"doublecheck_out/images"
LABELS_ROOT = r"doublecheck_out/labels"

SIZE_FALLBACK = (640, 640)

# Per-image bearing clustering (dedup within each final image if multiple boxes)
BEARING_EPS_DEG = 1.0     # sleeves very close in view => keep small (0.8–1.5)

# Triangulation gates (across different camera points)
PAIR_MIN_DIST_M       = 8.0     # minimum baseline (meters)
PAIR_MAX_DIST_M       = 120.0   # maximum baseline (meters)
MAX_PARALLEL_DEG      = 8.0     # skip near-parallel rays
FRONT_ONLY            = True    # intersection must be in front of both cameras
ANGULAR_RESID_MAX_DEG = 3.0     # max angular error to accept an intersection

# Only accept intersections that lie within this distance along each ray
# (matches mapping ray length intent)
INTERSECTION_MAX_RANGE_M = 30.0

# Single-ray fallback distance estimation (for rays without any crossing)
ASSUMED_SLEEVE_WIDTH_M = 0.20   # approximate physical sleeve width (meters)

# Merge nearby sleeve sites (post-cluster) so close sites become one bigger site
SITE_MERGE_EPS_M = 8.0          # meters

# Final site clustering (meters)
# Increase epsilon to merge near-duplicate intersections into one site
SITE_CLUSTER_EPS_M    = 5.0     # sleeves within 5 m considered same site
SITE_MIN_SAMPLES      = 1

# Output
OUT_CSV = "sleeves.csv"

# ====================== HELPERS =====================

def wrap_deg(a: float) -> float:
    a = a % 360.0
    return a + 360.0 if a < 0 else a

def parse_point_dir_name(dname: str) -> Tuple[float, float]:
    # "<index>_lat43.435485_lon-79.849288"
    mlat = re.search(r"lat(-?\d+(?:\.\d+)?)", dname)
    mlon = re.search(r"lon(-?\d+(?:\.\d+)?)", dname)
    if not mlat or not mlon:
        raise ValueError(f"Cannot parse lat/lon from folder: {dname}")
    return float(mlat.group(1)), float(mlon.group(1))

def parse_image_name(fname: str) -> Dict[str, float]:
    """
    Pull lat, lon, heading, pitch, fov, WxH from the filename.
    Works even with prefixes like 'dc_' and suffixes like '_final'.
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    def grab(tag, cast=float, required=True):
        m = re.search(rf"{tag}(-?\d+(?:\.\d+)?)", base)
        if not m:
            if required:
                raise ValueError(f"Missing {tag} in filename: {fname}")
            return None
        return cast(m.group(1))

    lat = grab("lat")
    # first 'lon' occurrence after 'lat'
    mlon = re.search(r"lon(-?\d+(?:\.\d+)?)", base)
    if not mlon:
        raise ValueError(f"Missing lon in filename: {fname}")
    lon = float(mlon.group(1))

    hdg = grab("hdg")
    pit = grab("pitch")
    fov = grab("fov")

    msize = re.search(r"_(\d+)x(\d+)", base)
    if msize:
        W, H = int(msize.group(1)), int(msize.group(2))
    else:
        W, H = SIZE_FALLBACK

    # optional snapped coordinates
    msnap = re.search(r"snap(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", base)
    snap_lat = float(msnap.group(1)) if msnap else None
    snap_lon = float(msnap.group(2)) if msnap else None

    return dict(lat=lat, lon=lon, snap_lat=snap_lat, snap_lon=snap_lon,
                heading=hdg, pitch=pit, fov=fov, W=W, H=H)

def bearing_from_bbox(xc_norm: float, hfov_deg: float) -> float:
    # Δheading = atan((2x-1)*tan(hFOV/2)) [deg] — perspective-correct
    u = 2.0 * xc_norm - 1.0
    hf = math.radians(hfov_deg)
    return math.degrees(math.atan(u * math.tan(hf / 2.0)))

def equirect_xy_m(lat0: float, lon0: float, lat: float, lon: float) -> Tuple[float, float]:
    R = 6371000.0
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y

def xy_to_latlon(lat0: float, lon0: float, x: float, y: float) -> Tuple[float, float]:
    R = 6371000.0
    lat = math.degrees(y / R) + lat0
    lon = math.degrees(x / (R * math.cos(math.radians(lat0)))) + lon0
    return lat, lon

def line_intersection_least_squares(P1, d1, P2, d2):
    """
    2D line (ray) intersection in least-squares sense.
    Returns X (closest point), t1, s2 or None if near-parallel.
    Lines are: P1 + t1*d1 and P2 + s2*d2
    """
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    P1 = np.asarray(P1, float); P2 = np.asarray(P2, float)
    A  = np.stack([d1, -d2], axis=1)  # (2,2)
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    ts = np.linalg.solve(A, (P2 - P1))
    t1, s2 = ts[0], ts[1]
    X = P1 + t1 * d1
    return X, t1, s2

def ang_between(v1, v2) -> float:
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 180.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def circular_mean_deg(angles_deg: List[float], weights: Optional[List[float]] = None) -> float:
    rads = np.radians(angles_deg)
    if weights is None:
        weights = np.ones_like(rads)
    C = np.sum(weights * np.cos(rads))
    S = np.sum(weights * np.sin(rads))
    return wrap_deg(math.degrees(math.atan2(S, C)))

def cluster_bearings(bearings_deg: List[float], weights: Optional[List[float]], eps_deg: float):
    """
    Circular clustering on [0,360). Returns [(mean_bearing, member_indices), ...]
    """
    if not bearings_deg:
        return []
    idxs = list(range(len(bearings_deg)))
    paired = sorted(zip(bearings_deg, idxs))
    angles = [a for a, _ in paired]; order = [i for _, i in paired]
    used = [False]*len(angles); clusters = []

    def within(a, b, eps):
        d = abs(a - b) % 360.0
        if d > 180.0: d = 360.0 - d
        return d <= eps

    for i, a in enumerate(angles):
        if used[i]: continue
        group = [i]; used[i] = True
        j = i + 1
        while j < len(angles) and within(angles[j], angles[j-1], eps_deg):
            group.append(j); used[j] = True; j += 1
        # wrap-around connect if first/last close
        if i == 0 and not used[-1] and within(angles[0], angles[-1], eps_deg):
            k = len(angles) - 1
            while k >= 0 and not used[k] and within(angles[(k+1)%len(angles)], angles[k], eps_deg):
                group.append(k); used[k] = True; k -= 1
        cluster_idxs = [order[g] for g in group]
        meanB = circular_mean_deg([angles[g] for g in group])
        clusters.append((meanB, cluster_idxs))
    return clusters

def iter_point_dirs(images_root: str):
    """Yield (route, point_dir, point_name) for images/<point> or images/<route>/<point>."""
    direct = [p for p in glob.glob(os.path.join(images_root, "*")) if os.path.isdir(p)]
    if any(re.search(r"lat-?\d", os.path.basename(p) or "") for p in direct):
        for pdir in sorted(direct):
            yield (None, pdir, os.path.basename(pdir))
        return
    for rdir in sorted([d for d in glob.glob(os.path.join(images_root, "*")) if os.path.isdir(d)]):
        route = os.path.basename(rdir)
        for pdir in sorted([p for p in glob.glob(os.path.join(rdir, "*")) if os.path.isdir(p)]):
            yield (route, pdir, os.path.basename(pdir))

# ====================== CORE ========================

def main():
    # Discover point folders across routes
    points = list(iter_point_dirs(IMAGES_ROOT))
    if not points:
        print("No point folders under", IMAGES_ROOT)
        return

    # Reference origin for local metric coords
    latlons = []
    for route, d, point_name in points:
        try:
            lat, lon = parse_point_dir_name(point_name)
            latlons.append((lat, lon))
        except Exception:
            pass
    if not latlons:
        print("Could not parse any lat/lon from point folders.")
        return
    lat0 = float(np.mean([p[0] for p in latlons]))
    lon0 = float(np.mean([p[1] for p in latlons]))

    # For each point folder: collect per-image bearings (per-image dedup), record camera XY
    # Camera XY uses snapped coordinates if available in filenames to match how rays are drawn.
    cams = []  # list of dict: {"x": Px, "y": Py, "bearings": [float,...]}
    for route, point_dir, point_name in points:
        img_files = sorted(glob.glob(os.path.join(point_dir, "*.jpg")))
        if not img_files:
            print(f"[{point_name}] no images.")
            continue

        lbl_dir = os.path.join(LABELS_ROOT, point_name) if route is None else os.path.join(LABELS_ROOT, route, point_name)
        if not os.path.isdir(lbl_dir):
            print(f"[WARN] No labels folder for {point_name}, skipping.")
            continue

        # Determine camera location for this point using snapped coords if present in filenames.
        # Fallback to folder lat/lon if no images contain snap tokens.
        cam_lats: List[float] = []
        cam_lons: List[float] = []

        per_point_bearings: List[float] = []
        det_total = 0
        dedup_total = 0

        for img_path in img_files:
            try:
                info = parse_image_name(img_path)
            except Exception as e:
                print(f"[{point_name}] {img_path}: {e}")
                continue

            # collect candidate camera coordinates from this image (prefer snapped)
            cam_lat_i = info["snap_lat"] if info.get("snap_lat") is not None else info["lat"]
            cam_lon_i = info["snap_lon"] if info.get("snap_lon") is not None else info["lon"]
            cam_lats.append(cam_lat_i)
            cam_lons.append(cam_lon_i)

            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                continue

            bearings_img: List[float] = []
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except Exception:
                continue

            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    xc = float(parts[1])
                except Exception:
                    continue

                dyaw = bearing_from_bbox(xc, info["fov"])
                beta = wrap_deg(info["heading"] + dyaw)
                bearings_img.append(beta)
                det_total += 1

            if bearings_img:
                clus = cluster_bearings(bearings_img, weights=None, eps_deg=BEARING_EPS_DEG)
                dedup_total += len(clus)
                for meanB, _ in clus:
                    per_point_bearings.append(meanB)

        if not per_point_bearings:
            print(f"[{point_name}] no detections.")
            continue

        # Across-image bearing dedup for this camera point
        cam_bearings: List[float] = []
        cl_cam = cluster_bearings(per_point_bearings, weights=None, eps_deg=BEARING_EPS_DEG)
        for meanB, _ in cl_cam:
            cam_bearings.append(meanB)

        if not cam_bearings:
            print(f"[{point_name}] no unique bearings after dedup.")
            continue

        # finalize camera location for this point
        if cam_lats and cam_lons:
            cam_lat = float(np.median(cam_lats))
            cam_lon = float(np.median(cam_lons))
        else:
            try:
                cam_lat, cam_lon = parse_point_dir_name(point_name)
            except Exception as e:
                print(f"[WARN] {point_name}: {e}; skipping.")
                continue

        Px, Py = equirect_xy_m(lat0, lon0, cam_lat, cam_lon)

        print(f"[{point_name}] camera=({cam_lat:.6f},{cam_lon:.6f}), "
              f"views={len(img_files)}, dets={det_total}, dedup_per_image_sum={dedup_total}, "
              f"unique_bearings_per_camera={len(cam_bearings)}")

        cams.append({"x": Px, "y": Py, "bearings": cam_bearings, "route": route if 'route' in locals() else None, "point_name": point_name})

    if not cams:
        print("No cameras with detections found.")
        return

    # Triangulate bearings from different cameras to produce sleeve XY points
    # Track which unique lines (camera index + bearing index) support each intersection.
    intersections = []  # list of dicts: {"xy": [x,y], "lines": set([(i,bi),(j,bj)])}
    for i in range(len(cams)):
        Pi = np.array([cams[i]["x"], cams[i]["y"]], float)
        for j in range(i + 1, len(cams)):
            Pj = np.array([cams[j]["x"], cams[j]["y"]], float)
            baseline = np.linalg.norm(Pi - Pj)
            if baseline < PAIR_MIN_DIST_M or baseline > PAIR_MAX_DIST_M:
                continue

            for bi, b_i in enumerate(cams[i]["bearings"]):
                di = np.array([math.sin(math.radians(b_i)),
                               math.cos(math.radians(b_i))], float)
                for bj, b_j in enumerate(cams[j]["bearings"]):
                    # skip near-parallel rays
                    diff = abs(((b_i - b_j + 180) % 360) - 180)
                    if diff < MAX_PARALLEL_DEG:
                        continue
                    dj = np.array([math.sin(math.radians(b_j)),
                                   math.cos(math.radians(b_j))], float)

                    res = line_intersection_least_squares(Pi, di, Pj, dj)
                    if res is None:
                        continue
                    X, t1, s2 = res

                    if FRONT_ONLY and (t1 <= 0 or s2 <= 0):
                        continue

                    # range gate: intersections must be within max range along both rays
                    if (t1 > INTERSECTION_MAX_RANGE_M) or (s2 > INTERSECTION_MAX_RANGE_M):
                        continue

                    # angular residual gate
                    if ang_between(di, X - Pi) > ANGULAR_RESID_MAX_DEG:  continue
                    if ang_between(dj, X - Pj) > ANGULAR_RESID_MAX_DEG:  continue

                    intersections.append({
                        "xy": [float(X[0]), float(X[1])],
                        "lines": {(i, bi), (j, bj)}
                    })

    # If no intersections, we will proceed to approximate sleeves from single rays

    # Build sleeves list: intersections + single-ray estimates for unused bearings
    sleeves_xy = [it["xy"] for it in intersections]

    # Build used bearings index per camera
    used_map = defaultdict(set)
    for it in intersections:
        for (ci, bi) in it.get("lines", set()):
            used_map[ci].add(bi)

    # For each camera, for bearings not used, approximate distance from label widths at that point
    for idx_cam, cam in enumerate(cams):
        # Collect width stats from this point's labels to estimate distance
        route = cam.get("route")
        point_name = cam.get("point_name")
        lbl_dir = os.path.join(LABELS_ROOT, point_name) if not route else os.path.join(LABELS_ROOT, route, point_name)
        widths = []
        hfovs = []
        try:
            for txt in glob.glob(os.path.join(lbl_dir, "*_final.txt")):
                base = os.path.splitext(os.path.basename(txt))[0]
                info = parse_image_name(base)
                hfovs.append(float(info.get("fov", 60.0)))
                with open(txt, "r", encoding="utf-8") as f:
                    for ln in f:
                        ps = ln.strip().split()
                        if len(ps) >= 5:
                            try:
                                widths.append(float(ps[3]))
                            except Exception:
                                pass
        except Exception:
            pass
        if not widths:
            est_dist = INTERSECTION_MAX_RANGE_M
        else:
            w_med = float(np.median(widths))
            hfov_med = float(np.median(hfovs)) if hfovs else 60.0
            theta = 2.0 * math.atan(max(1e-6, w_med) * math.tan(math.radians(hfov_med/2.0)))
            est = ASSUMED_SLEEVE_WIDTH_M / (2.0 * math.tan(theta/2.0)) if theta > 1e-6 else INTERSECTION_MAX_RANGE_M
            est_dist = max(2.0, min(INTERSECTION_MAX_RANGE_M, est))

        Pi = np.array([cam["x"], cam["y"]], float)
        for bi, b in enumerate(cam["bearings"]):
            if bi in used_map.get(idx_cam, set()):
                continue
            dvec = np.array([math.sin(math.radians(b)), math.cos(math.radians(b))], float)
            Xs = Pi + dvec * est_dist
            sleeves_xy.append([float(Xs[0]), float(Xs[1])])

    if not sleeves_xy:
        print("No sleeves found (no intersections and no single-ray estimates).")
        return

    # Cluster sleeves to sites
    X = np.array(sleeves_xy, float)
    clustering = DBSCAN(eps=SITE_CLUSTER_EPS_M, min_samples=SITE_MIN_SAMPLES).fit(X)
    labels = clustering.labels_

    # accumulate per-site points
    initial_sites: List[List[List[float]]] = []
    tmp_sites: Dict[int, List[List[float]]] = defaultdict(list)
    for k, lab in enumerate(labels):
        lab_id = lab if lab != -1 else (-100000 - k)
        tmp_sites[lab_id].append([float(X[k,0]), float(X[k,1])])
    for _, pts in tmp_sites.items():
        initial_sites.append(pts)

    # Merge sites within SITE_MERGE_EPS_M (greedy agglomeration on site centroids)
    centroids = []
    for pts in initial_sites:
        arr = np.array(pts, float)
        centroids.append(np.array([float(np.median(arr[:,0])), float(np.median(arr[:,1]))], float))

    n = len(initial_sites)
    visited = [False]*n
    merged_sites: List[List[List[float]]] = []
    for i in range(n):
        if visited[i]:
            continue
        group_idx = [i]
        visited[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if visited[j]:
                    continue
                # min distance to any member centroid in group
                dmin = min(np.linalg.norm(centroids[j] - centroids[g]) for g in group_idx)
                if dmin <= SITE_MERGE_EPS_M:
                    visited[j] = True
                    group_idx.append(j)
                    changed = True
        # merge all points from grouped sites
        merged_pts: List[List[float]] = []
        for g in group_idx:
            merged_pts.extend(initial_sites[g])
        merged_sites.append(merged_pts)

    # Summarize sites -> rows (lat, lon, number)
    rows = []
    for pts in merged_sites:
        pts_np = np.array(pts, float)
        mx, my = float(np.median(pts_np[:, 0])), float(np.median(pts_np[:, 1]))
        lat, lon = xy_to_latlon(lat0, lon0, mx, my)
        # Keep count internally for sizing on the map; map code suppresses numeric labels
        rows.append({"lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "number": len(pts)})

    rows.sort(key=lambda r: (r["lat"], r["lon"]))

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lat", "lon", "number"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {OUT_CSV} with {len(rows)} site(s). Example rows:")
    for r in rows[:5]:
        print(f"  {r['lat']},{r['lon']},{r['number']}")

if __name__ == "__main__":
    main()
