#!/usr/bin/env python3
"""
Street View 360° sweep along a line between two coordinates (SNAP mode).

- Hardcoded START and END coordinates
- Sample points every STEP_METERS along the straight line
- At each point, SNAPS to nearest pano using 'location=lat,lng' (no 'pano=')
- Saves a 360° sweep (0..360 in STEP_DEG)
- Writes points.csv with all sampled (original) points

Folder layout (unchanged except base folder moved under route_1):
  sweep_out/images/route_1/<index>_lat<lat>_lon<lon>/
    sweep_lat<orig_lat>_lon<orig_lon>_snap<slat>_<slon>_hdgXXX_pitchP_fovF_WxH.jpg
"""

import os
import csv
import time
import math
import requests
import concurrent.futures

# ===================== CONFIG =====================

API_KEY  = "AIzaSyAal2PH1aIbKywxU2JbjxNRIoBrSPBXinY"  # <-- replace

# Start & destination
START_LAT = 43.437807
START_LON = -79.855767
END_LAT   = 43.434636
END_LON   = -79.847378

# Sampling along the path
STEP_METERS = 10.0      # distance between points along the line
INCLUDE_ENDPOINT = True # also include the exact END point

# Sweep params
# Heading step in degrees (captures 0, step, 2*step, ... < 360)
STEP_DEG = 60.0
FOV      = 60.0
# Capture multiple camera pitches (degrees). Positive looks up, negative looks down.
PITCHES  = [15.0, 75.0]
SIZE     = (640, 640)   # (W,H)
RADIUS   = 50           # used only for metadata snapping (optional)
SLEEP    = 0.05         # pause between HTTP requests (sec)

# Concurrency controls
MAX_WORKERS = 12        # parallel image requests per point
REQUEST_TIMEOUT = 30    # seconds per image request

# Output
OUT_DIR_IMAGES = "sweep_out/images/route_1"
OUT_POINTS_CSV = "points.csv"
PER_POINT_SUBDIR = True

# ===================== HELPERS =====================

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))

def interpolate_points_by_step(start_lat, start_lon, end_lat, end_lon, step_m: float, include_endpoint: bool=True):
    """Return (lat, lon) points every ~step_m between start and end (plus endpoints)."""
    total = haversine_m(start_lat, start_lon, end_lat, end_lon)
    if total == 0.0:
        return [(round(start_lat,7), round(start_lon,7))] if not include_endpoint else [
            (round(start_lat,7), round(start_lon,7)),
            (round(end_lat,7), round(end_lon,7))
        ]

    n_steps = max(1, int(math.floor(total / step_m)))
    pts = []
    for i in range(n_steps + 1):  # include start and the last interior step
        t = i / n_steps
        lat = start_lat + (end_lat - start_lat) * t
        lon = start_lon + (end_lon - start_lon) * t
        pts.append((round(lat, 7), round(lon, 7)))

    if include_endpoint:
        end_pair = (round(end_lat,7), round(end_lon,7))
        if pts[-1] != end_pair:
            pts.append(end_pair)
    return pts

def headings_list(step_deg: float):
    headings = []
    a = 0.0
    while a < 360.0 - 1e-6:
        headings.append(round(a, 6))
        a += step_deg
    return headings

def streetview_metadata_snap(lat: float, lon: float, api_key: str, radius: int = 50):
    """
    Optional: use metadata to learn the snapped location.
    We still request images with 'location=snapped_lat,snapped_lon' (no 'pano=').
    """
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": radius, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "OK":
            loc = j.get("location", {})
            return float(loc.get("lat", lat)), float(loc.get("lng", lon))
    except Exception:
        pass
    # fallback to original if metadata fails
    return lat, lon

def streetview_image_location(lat: float, lon: float, heading: float, pitch: float, fov: float,
                              size_wh: tuple[int,int], api_key: str) -> bytes:
    """
    Fetch image using 'location=lat,lon' (snaps to nearest pano). No 'pano=' used.
    """
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": f"{size_wh[0]}x{size_wh[1]}",
        "location": f"{lat:.7f},{lon:.7f}",
        "heading": f"{heading:.6f}",
        "pitch": f"{pitch:.6f}",
        "fov": f"{fov:.6f}",
        "key": api_key,
    }
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def fetch_and_save_image(snap_lat: float, snap_lon: float, heading: float, pitch: float,
                         fov: float, size_wh: tuple[int,int], api_key: str, out_path: str) -> tuple[bool, str, str]:
    """Fetch one Street View image and write to out_path. Returns (ok, out_path, err_msg)."""
    try:
        img_bytes = streetview_image_location(snap_lat, snap_lon, heading, pitch, fov, size_wh, api_key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return True, out_path, ""
    except Exception as e:
        return False, out_path, str(e)

# ===================== MAIN =====================

def main():
    os.makedirs(OUT_DIR_IMAGES, exist_ok=True)
    pts = interpolate_points_by_step(START_LAT, START_LON, END_LAT, END_LON, STEP_METERS, INCLUDE_ENDPOINT)
    # Headings list from a single step size
    hdgs = headings_list(STEP_DEG)

    # Write points.csv (lat,lon) — original sampling points
    with open(OUT_POINTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon"])
        for (la, lo) in pts:
            w.writerow([f"{la:.7f}", f"{lo:.7f}"])
    print(f"[INFO] Wrote {OUT_POINTS_CSV} with {len(pts)} points.")

    # Sweep each point (SNAP mode)
    for idx, (lat, lon) in enumerate(pts):
        # (Optional) get the snapped location for consistent headings across the sweep
        snap_lat, snap_lon = streetview_metadata_snap(lat, lon, API_KEY, radius=RADIUS)
        print(f"[{idx:05d}] snap=({snap_lat:.6f},{snap_lon:.6f}) from orig=({lat:.6f},{lon:.6f})")

        # per-point subfolder (same pattern as before)
        dest_dir = OUT_DIR_IMAGES
        if PER_POINT_SUBDIR:
            dest_dir = os.path.join(OUT_DIR_IMAGES, f"{idx:05d}_lat{lat:.6f}_lon{lon:.6f}")
            os.makedirs(dest_dir, exist_ok=True)

        # save sweep images using 'location=snapped coords' for each requested pitch, in parallel
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for p_idx, pitch in enumerate(PITCHES):
                for j, hdg in enumerate(hdgs):
                    out_name = (f"{dest_dir}/sweep_lat{lat:.6f}_lon{lon:.6f}"
                                f"_snap{snap_lat:.6f}_{snap_lon:.6f}"
                                f"_hdg{int(hdg):03d}_pitch{int(pitch)}"
                                f"_fov{int(FOV)}_{SIZE[0]}x{SIZE[1]}.jpg")
                    futures.append(ex.submit(
                        fetch_and_save_image, snap_lat, snap_lon, hdg, pitch, FOV, SIZE, API_KEY, out_name
                    ))

            for fut in concurrent.futures.as_completed(futures):
                ok, path, err = fut.result()
                if ok:
                    print(f"[OK] Saved {path}")
                else:
                    print(f"[ERR] {path}: {err}")

    print("Done.")

if __name__ == "__main__":
    main()
