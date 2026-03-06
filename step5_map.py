#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map_quicksleeves.py

Builds an interactive HTML map from doublecheck_out:
- One marker per camera point (final zoom images).
- For each detected quick sleeve in a final image, a short line showing its direction from the camera.
- Uses snapped coordinates (snap<lat>_<lon>) as the camera location if present in the filename.
- NEW: If sleeves.csv exists (produced by sleeve_counter.py), overlay sleeve site markers labeled
  by count (handles cases where 3+ rays don't intersect exactly; sleeve_counter clusters with deviation).

Input tree:
  doublecheck_out/
    images/<POINT>/*_final.jpg
    labels/<POINT>/*_final.txt

Image filename pattern (tokens can be anywhere in the name):
  ... lat{LAT}_lon{LON}_snap{SLAT}_{SLON}_hdg{H}_pitch{P}_fov{F}_{W}x{H}_final.jpg

Label format per line:
  <cls> <xc> <yc> <w> <h> [conf]   (normalized coords; conf optional)

Optional overlay input:
  sleeves.csv  (lat,lon,number)  from sleeve_counter.py

Output:
  sleeve_directions_map.html
"""

import os
import re
import csv
import glob
import math
from typing import Dict, Tuple, List, Optional

import folium
from folium.features import DivIcon

# ============== CONFIG ==============

IMAGES_ROOT = r"doublecheck_out/images"  # not strictly needed; kept for reference
LABELS_ROOT = r"doublecheck_out/labels"  # primary source for rays (final txts)

# How long to draw the direction rays (meters)
RAY_LENGTH_M = 35.0

# If size WxH isn't found in filename, use this
SIZE_FALLBACK = (640, 640)

# If you want to thin the rays (e.g., only draw top-N per image), set N>0; else 0 keeps all
TOP_N_PER_IMAGE = 0  # 0 = keep all detections

# Map output
OUT_HTML = "sleeve_directions_map4.html"
SLEEVES_CSV = "sleeves.csv"

# Ray de-duplication within the same position folder (point)
# If two or more rays have bearings within this epsilon (deg), keep one.
RAY_DEDUP_EPS_DEG = 3.0

# ============== GEO/PROJ HELPERS ==============

EARTH_R = 6371000.0

def dest_point(lat: float, lon: float, bearing_deg: float, dist_m: float) -> Tuple[float, float]:
    """Forward geodesic: destination point from lat/lon, bearing (deg), distance (m)."""
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = dist_m / EARTH_R

    lat2 = math.asin(math.sin(lat1)*math.cos(dr) + math.cos(lat1)*math.sin(dr)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return brng

def bearing_from_bbox_x(xc_norm: float, hfov_deg: float) -> float:
    """Perspective-correct horizontal angle offset: Δheading = atan((2x-1)*tan(hFOV/2))."""
    u = 2.0 * xc_norm - 1.0
    hf = math.radians(hfov_deg)
    return math.degrees(math.atan(u * math.tan(hf / 2.0)))

def parse_tokens_from_name(fname: str) -> Dict[str, Optional[float]]:
    """
    Extract tokens from filename:
      lat, lon, (optional) snap_lat/snap_lon, hdg, pitch, fov, W, H.
    Works with names like:
      dc_sweep_lat43.436639_lon-79.852676_snap43.436948_-79.852423_hdg270_pitch0_fov60_640x640_det0_final.jpg
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
    m_lon = re.search(r"lon(-?\d+(?:\.\d+)?)", base)
    if not m_lon:
        raise ValueError(f"Missing lon in filename: {fname}")
    lon = float(m_lon.group(1))

    # Optional snapped coordinates
    m_snap = re.search(r"snap(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", base)
    snap_lat = float(m_snap.group(1)) if m_snap else None
    snap_lon = float(m_snap.group(2)) if m_snap else None

    hdg = grab("hdg")
    pit = grab("pitch")
    fov = grab("fov")

    msize = re.search(r"_(\d+)x(\d+)", base)
    if msize:
        W, H = int(msize.group(1)), int(msize.group(2))
    else:
        W, H = SIZE_FALLBACK

    return dict(lat=lat, lon=lon, snap_lat=snap_lat, snap_lon=snap_lon,
                heading=hdg, pitch=pit, fov=fov, W=W, H=H)

# ============== MAIN ==============

def iter_point_label_dirs(labels_root: str):
    # Yield (route, point_dir, point_name). Supports labels/<point> or labels/<route>/<point>.
    direct_points = [p for p in glob.glob(os.path.join(labels_root, "*")) if os.path.isdir(p)]
    # If direct points contain txt files, treat as points; else check for routes
    has_direct = any(glob.glob(os.path.join(p, "*.txt")) for p in direct_points)
    if has_direct:
        for pdir in sorted(direct_points):
            yield (None, pdir, os.path.basename(pdir))
        return
    # else assume routes
    for rdir in sorted([d for d in glob.glob(os.path.join(labels_root, "*")) if os.path.isdir(d)]):
        route = os.path.basename(rdir)
        for pdir in sorted([p for p in glob.glob(os.path.join(rdir, "*")) if os.path.isdir(p)]):
            yield (route, pdir, os.path.basename(pdir))


def main():
    # Collect final label files under labels/ (across routes) and build rays
    camera_points: List[Tuple[float, float]] = []  # for map centering
    all_rays: List[Dict] = []
    total_txt = 0
    total_rays = 0

    # Group rays by point to enable per-point deduplication
    rays_by_point: Dict[Tuple[Optional[str], str], List[Dict]] = {}

    any_points = False
    for route, pdir, point_name in iter_point_label_dirs(LABELS_ROOT):
        any_points = True
        txt_files = sorted(glob.glob(os.path.join(pdir, "*_final.txt")))
        if not txt_files:
            continue
        # Prefer cluster-based finals if present for this point; otherwise include all finals
        cluster_files = [p for p in txt_files if "_cluster" in os.path.basename(p)]
        if cluster_files:
            txt_files = cluster_files
        key = (route, point_name)
        rays_by_point[key] = []

        for lbl_path in txt_files:
            total_txt += 1
            base = os.path.splitext(os.path.basename(lbl_path))[0]
            # Infer imaging info directly from filename tokens
            try:
                info = parse_tokens_from_name(base)
            except Exception as e:
                # Try parsing from a full image path if needed
                try:
                    info = parse_tokens_from_name(lbl_path)
                except Exception:
                    print(f"[WARN] {lbl_path}: {e}")
                    continue

            cam_lat = info["snap_lat"] if info["snap_lat"] is not None else info["lat"]
            cam_lon = info["snap_lon"] if info["snap_lon"] is not None else info["lon"]
            camera_points.append((cam_lat, cam_lon))

            # Read label lines
            dets = []
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        parts = ln.split()
                        if len(parts) < 5:
                            continue
                        try:
                            xc = float(parts[1])
                            conf = float(parts[5]) if len(parts) >= 6 else 0.0
                            dets.append((xc, conf))
                        except Exception:
                            continue
            except Exception:
                continue

            if TOP_N_PER_IMAGE > 0 and len(dets) > TOP_N_PER_IMAGE:
                dets = sorted(dets, key=lambda t: t[1], reverse=True)[:TOP_N_PER_IMAGE]

            for (xc, conf) in dets:
                dyaw = bearing_from_bbox_x(xc, info["fov"])
                abs_bearing = (info["heading"] + dyaw) % 360.0
                to_lat, to_lon = dest_point(cam_lat, cam_lon, abs_bearing, RAY_LENGTH_M)
                rays_by_point[key].append({
                    "cam_lat": cam_lat,
                    "cam_lon": cam_lon,
                    "to_lat": to_lat,
                    "to_lon": to_lon,
                    "bearing": abs_bearing,
                    "conf": conf,
                    "img": base + ".jpg",
                    "used_snap": info["snap_lat"] is not None
                })
                total_rays += 1

    if not any_points:
        print("No label point folders found under", LABELS_ROOT)
        return

    # De-duplicate rays per point by bearing proximity
    def dedup_rays(rlist: List[Dict]) -> List[Dict]:
        if not rlist:
            return rlist
        kept: List[Dict] = []
        bearings: List[float] = []
        for r in sorted(rlist, key=lambda x: (-x.get("conf", 0.0), x["bearing"])):
            b = r["bearing"]
            dup = False
            for bb in bearings:
                d = abs(((b - bb + 180.0) % 360.0) - 180.0)
                if d <= RAY_DEDUP_EPS_DEG:
                    dup = True
                    break
            if not dup:
                kept.append(r)
                bearings.append(b)
        return kept

    deduped_rays: List[Dict] = []
    for key, rlist in rays_by_point.items():
        deduped_rays.extend(dedup_rays(rlist))

    # Prepare map center
    if camera_points:
        lats = sorted([p[0] for p in camera_points])
        lons = sorted([p[1] for p in camera_points])
        center_lat = lats[len(lats)//2]
        center_lon = lons[len(lons)//2]
    else:
        center_lat, center_lon = 43.0, -79.0

    if not deduped_rays and not camera_points:
        print("No final images or labels found under doublecheck_out.")
        return

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="OpenStreetMap")

    # Feature layers for toggling
    fg_cams = folium.FeatureGroup(name="Cameras", show=True)
    fg_rays = folium.FeatureGroup(name="Rays", show=True)
    fg_sites = folium.FeatureGroup(name="Sleeve Sites", show=True)

    # Add camera markers
    for (lat, lon) in camera_points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2A93EE",
            weight=2,
            fill=True,
            fill_opacity=0.9,
            popup=f"Camera: {lat:.6f},{lon:.6f}"
        ).add_to(fg_cams)

    # Add rays
    for r in deduped_rays:
        tip = f"{r['img']}  bearing={r['bearing']:.1f}°, conf={r['conf']:.2f}"
        if r["used_snap"]:
            tip += " (snap)"
        folium.PolyLine(
            locations=[[r["cam_lat"], r["cam_lon"]], [r["to_lat"], r["to_lon"]]],
            weight=3,
            color="#FF5722",
            opacity=0.9,
            tooltip=tip
        ).add_to(fg_rays)

    # Optional: overlay sleeve sites from sleeves.csv (if present)
    sites_count = 0
    if os.path.exists(SLEEVES_CSV):
        try:
            with open(SLEEVES_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        slat = float(row.get("lat", ""))
                        slon = float(row.get("lon", ""))
                    except Exception:
                        continue

                    # number is optional; if missing/blank, draw without label and default radius
                    raw_num = (row.get("number", "") or "").strip()
                    num_val = None
                    try:
                        if raw_num != "":
                            num_val = int(float(raw_num))
                    except Exception:
                        num_val = None

                    # Always suppress numeric labels; use count only to scale radius
                    radius = 6 if num_val is None else 4 + min(12, max(0, num_val))
                    tip = "Sleeve site"

                    # Find nearest camera point for Street View viewpoint and compute heading to site
                    sv_lat, sv_lon = slat, slon  # fallback viewpoint = site
                    heading = None
                    if camera_points:
                        # nearest camera by haversine approximation (equirect)
                        best_d2 = None
                        for (clat, clon) in camera_points:
                            # quick planar approximation in meters
                            x = math.radians(slon - clon) * EARTH_R * math.cos(math.radians((slat + clat) / 2.0))
                            y = math.radians(slat - clat) * EARTH_R
                            d2 = x*x + y*y
                            if best_d2 is None or d2 < best_d2:
                                best_d2 = d2
                                sv_lat, sv_lon = clat, clon
                        heading = initial_bearing_deg(sv_lat, sv_lon, slat, slon)
                        # If camera is extremely close to the site, back off a few meters to avoid odd Street View init
                        try:
                            dist_m = math.sqrt(best_d2) if best_d2 is not None else 0.0
                            if dist_m < 2.0:
                                sv_lat, sv_lon = dest_point(sv_lat, sv_lon, (heading + 180.0) % 360.0, 5.0)
                        except Exception:
                            pass

                    # Build a Google Maps Street View URL
                    pitch_param = -10  # slight downward tilt to avoid black screen on init
                    if heading is None:
                        gmaps_url = ("https://www.google.com/maps/@?api=1&map_action=pano"
                                     f"&viewpoint={slat:.6f},{slon:.6f}&pitch={pitch_param}&fov=80")
                    else:
                        gmaps_url = ("https://www.google.com/maps/@?api=1&map_action=pano"
                                     f"&viewpoint={sv_lat:.6f},{sv_lon:.6f}&heading={heading:.0f}&pitch={pitch_param}&fov=80")

                    folium.CircleMarker(
                        location=[slat, slon],
                        radius=radius,
                        color="#00C853",
                        fill=True,
                        fill_color="#00C853",
                        fill_opacity=0.9,
                        weight=2,
                        tooltip=tip,
                        popup=folium.Popup(
                            html=f'<a href="{gmaps_url}" target="_blank">Open Street View →</a>',
                            max_width=250
                        )
                    ).add_to(fg_sites)

                    # Do not draw numeric labels on sites
                    sites_count += 1
        except Exception as e:
            print(f"[WARN] Failed to read {SLEEVES_CSV}: {e}")
    else:
        print(f"[INFO] {SLEEVES_CSV} not found; sleeve site markers skipped.")

    # Add layers to map and controls
    fg_cams.add_to(m)
    fg_rays.add_to(m)
    fg_sites.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(OUT_HTML)
    print(f"Map written to {OUT_HTML}")
    print(f"Camera points: {len(camera_points)}  |  Rays (raw): {total_rays}  |  Rays drawn (dedup): {len(deduped_rays)}  |  Label files: {total_txt}")
    if os.path.exists(SLEEVES_CSV):
        print(f"Sleeve sites plotted from {SLEEVES_CSV}: {sites_count}")

if __name__ == "__main__":
    main()
