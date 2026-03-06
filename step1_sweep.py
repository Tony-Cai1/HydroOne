#!/usr/bin/env python3
"""
Street View 360° sweep sampled along a Google Directions route SNAP mode.
"""

import os
import csv
import time
import math
import requests
import concurrent.futures
from typing import List, Tuple, Optional

# load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ============== CONFIG ==============

API_KEY = "AIzaSyAal2PH1aIbKywxU2JbjxNRIoBrSPBXinY"

START_LAT, START_LON= 45.333399, -79.223486
END_LAT, END_LON    = 45.339132, -79.220991

USE_DIRECTIONS = True
TRAVEL_MODE    = "driving"
AVOID          = None
DEPARTURE_TIME = "now"
WAYPOINTS: Optional[List[Tuple[float, float]]] = None

STEP_METERS = 10.0
INCLUDE_ENDPOINT = True

STEP_DEG = 60.0
FOV      = 60.0
PITCHES  = [15.0, 75.0]
SIZE     = (640, 640)
RADIUS   = 50
SLEEP    = 0.02

MAX_WORKERS = 8                 # slightly lower to avoid overwhelming
REQUEST_TIMEOUT = 15            # stricter timeout per HTTP request
POINTS_HARD_CAP = 5000          # safety cap on number of sampled points

OUT_DIR_IMAGES = "sweep_out/images/route_5"
OUT_POINTS_CSV = "points.csv"
PER_POINT_SUBDIR = True

DEBUG = True
def dbg(*a):
    if DEBUG:
        print("[DBG]", *a, flush=True)

# ============== HELPERS ==============

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))

def headings_list(step_deg: float) -> List[float]:
    headings = []
    a = 0.0
    while a < 360.0 - 1e-6:
        headings.append(round(a, 6))
        a += step_deg
    return headings

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    points = []
    index = lat = lng = 0
    length = len(polyline_str)

    while index < length:
        shift = result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        points.append((lat / 1e5, lng / 1e5))
    return points

def resample_path(coords: List[Tuple[float, float]], step_m: float, include_endpoint: bool=True) -> List[Tuple[float, float]]:
    """
    Robust resampling: adds a hard iteration cap to avoid rare infinite loops
    when input polylines contain pathological repeated or jittery points.
    """
    if not coords:
        return []
    if len(coords) == 1:
        return [(round(coords[0][0], 7), round(coords[0][1], 7))]

    out: List[Tuple[float, float]] = []
    seg_idx = 0
    curr = coords[0]
    out.append((round(curr[0], 7), round(curr[1], 7)))
    dist_to_next = max(0.1, float(step_m))  # guard against zero/negative
    max_iters = (len(coords) * 1000)  # generous but finite
    iters = 0

    while seg_idx < len(coords) - 1:
        iters += 1
        if iters > max_iters:
            dbg("resample_path hit iteration cap; breaking early")
            break

        a = curr
        b = coords[seg_idx + 1]
        seg_len = haversine_m(a[0], a[1], b[0], b[1])

        # collapse degenerate segments
        if seg_len < 1e-6:
            seg_idx += 1
            curr = b
            continue

        if seg_len + 1e-9 >= dist_to_next:
            t = dist_to_next / seg_len
            lat = a[0] + (b[0] - a[0]) * t
            lon = a[1] + (b[1] - a[1]) * t
            curr = (lat, lon)
            quant = (round(lat, 7), round(lon, 7))
            if quant != out[-1]:
                out.append(quant)
            dist_to_next = step_m
        else:
            dist_to_next -= seg_len
            seg_idx += 1
            curr = b

        if len(out) >= POINTS_HARD_CAP:
            dbg(f"Resample produced >= {POINTS_HARD_CAP} points; truncating")
            break

    if include_endpoint:
        end_lat, end_lon = coords[-1]
        end_pair = (round(end_lat, 7), round(end_lon, 7))
        if out[-1] != end_pair:
            out.append(end_pair)
    return out

def build_waypoints_param(waypoints: Optional[List[Tuple[float, float]]]) -> Optional[str]:
    if not waypoints:
        return None
    return "|".join([f"{lat:.7f},{lon:.7f}" for lat, lon in waypoints])

def fetch_directions_polyline(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    api_key: str,
    mode: str = "driving",
    avoid: Optional[str] = None,
    departure_time: Optional[str] = None,
    waypoints: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is missing")

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat:.7f},{start_lon:.7f}",
        "destination": f"{end_lat:.7f},{end_lon:.7f}",
        "mode": mode,
        "key": api_key,
        "alternatives": "false",
    }
    if avoid:
        params["avoid"] = avoid
    if departure_time:
        params["departure_time"] = departure_time
    wp = build_waypoints_param(waypoints)
    if wp:
        params["waypoints"] = wp

    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    status = j.get("status")
    if status != "OK":
        raise RuntimeError(f"Directions API error: {status} details={j.get('error_message')}")

    routes = j.get("routes", [])
    if not routes:
        raise RuntimeError("No routes returned from Directions API")

    poly = routes[0].get("overview_polyline", {}).get("points")
    if not poly:
        raise RuntimeError("No overview_polyline in route")

    return decode_polyline(poly)

def streetview_metadata_snap(lat: float, lon: float, api_key: str, radius: int = 50) -> Tuple[float, float]:
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is missing")
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": radius, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "OK":
            loc = j.get("location", {})
            return float(loc.get("lat", lat)), float(loc.get("lng", lon))
    except Exception as e:
        dbg("metadata snap failed:", e)
    return lat, lon

def streetview_image_location(lat: float, lon: float, heading: float, pitch: float, fov: float,
                              size_wh: Tuple[int, int], api_key: str) -> bytes:
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is missing")
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
                         fov: float, size_wh: Tuple[int, int], api_key: str, out_path: str) -> Tuple[bool, str, str]:
    try:
        img_bytes = streetview_image_location(snap_lat, snap_lon, heading, pitch, fov, size_wh, api_key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return True, out_path, ""
    except Exception as e:
        return False, out_path, str(e)

def interpolate_points_by_step(start_lat, start_lon, end_lat, end_lon, step_m: float, include_endpoint: bool=True):
    total = haversine_m(start_lat, start_lon, end_lat, end_lon)
    if total == 0.0:
        return [(round(start_lat,7), round(start_lon,7))] if not include_endpoint else [
            (round(start_lat,7), round(start_lon,7)),
            (round(end_lat,7), round(end_lon,7))
        ]
    n_steps = max(1, int(math.floor(total / step_m)))
    pts = []
    for i in range(n_steps + 1):
        t = i / n_steps
        lat = start_lat + (end_lat - start_lat) * t
        lon = start_lon + (end_lon - start_lon) * t
        pts.append((round(lat, 7), round(lon, 7)))
    if include_endpoint:
        end_pair = (round(end_lat,7), round(end_lon,7))
        if pts[-1] != end_pair:
            pts.append(end_pair)
    return pts

# ============== MAIN ==============

def main():
    if not API_KEY or API_KEY == "YOUR_KEY_HERE":
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set")

    os.makedirs(OUT_DIR_IMAGES, exist_ok=True)

    if USE_DIRECTIONS:
        print("[INFO] Fetching route from Directions API", flush=True)
        route_coords = fetch_directions_polyline(
            START_LAT, START_LON, END_LAT, END_LON,
            api_key=API_KEY,
            mode=TRAVEL_MODE, avoid=AVOID,
            departure_time=DEPARTURE_TIME, waypoints=WAYPOINTS
        )
        print(f"[INFO] Route vertices: {len(route_coords)}", flush=True)
        dbg("First 3 vertices:", route_coords[:3])
        dbg("Last 3 vertices:", route_coords[-3:])
        pts = resample_path(route_coords, STEP_METERS, include_endpoint=INCLUDE_ENDPOINT)
        print(f"[INFO] Resampled {len(pts)} points at ~{STEP_METERS} m spacing", flush=True)
    else:
        print("[INFO] Using straight line interpolation", flush=True)
        pts = interpolate_points_by_step(START_LAT, START_LON, END_LAT, END_LON, STEP_METERS, INCLUDE_ENDPOINT)
        print(f"[INFO] Straight line points: {len(pts)}", flush=True)

    if len(pts) > POINTS_HARD_CAP:
        dbg(f"Clipping points from {len(pts)} to {POINTS_HARD_CAP}")
        pts = pts[:POINTS_HARD_CAP]

    # Write points.csv
    with open(OUT_POINTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon"])
        for la, lo in pts:
            w.writerow([f"{la:.7f}", f"{lo:.7f}"])
    print(f"[INFO] Wrote {OUT_POINTS_CSV} with {len(pts)} points", flush=True)

    hdgs = headings_list(STEP_DEG)

    # Sweep each point
    for idx, (lat, lon) in enumerate(pts):
        snap_lat, snap_lon = streetview_metadata_snap(lat, lon, API_KEY, radius=RADIUS)
        print(f"[{idx:05d}] snap=({snap_lat:.6f},{snap_lon:.6f}) from orig=({lat:.6f},{lon:.6f})", flush=True)

        dest_dir = OUT_DIR_IMAGES
        if PER_POINT_SUBDIR:
            dest_dir = os.path.join(OUT_DIR_IMAGES, f"{idx:05d}_lat{lat:.6f}_lon{lon:.6f}")
            os.makedirs(dest_dir, exist_ok=True)

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for pitch in PITCHES:
                for hdg in hdgs:
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
                    print(f"[OK] Saved {path}", flush=True)
                else:
                    print(f"[ERR] {path}: {err}", flush=True)

        time.sleep(SLEEP)

    print("Done.", flush=True)

if __name__ == "__main__":
    main()
