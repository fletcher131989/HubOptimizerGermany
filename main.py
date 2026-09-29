from functools import lru_cache
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree
import folium

_HERE = Path(__file__).parent
DATA_FOLDER = _HERE / "raw_german_postcode_data"
EARTH_RADIUS_KM = 6371.0
KM_PER_MILE = 1.60934
_PARQUET_COLS = ["postcode", "lat", "lon", "population"]
HUB_COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "cadetblue", "darkgreen", "darkpurple",
    "lightred", "beige", "darkblue", "lightblue", "lightgreen",
    "gray", "black",
]


# --------------------------------------------------
# Distance calculations
# --------------------------------------------------

@lru_cache(maxsize=None)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def haversine_array(lat1, lon1, lats2, lons2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lats2 = np.radians(lats2)
    lons2 = np.radians(lons2)

    dlat = lats2 - lat1
    dlon = lons2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def convert_to_km(distance, unit):
    unit = unit.lower()

    if unit in ["km", "kilometers", "kilometres"]:
        return distance

    if unit in ["mile", "miles", "mi"]:
        return distance * KM_PER_MILE

    raise ValueError("Unit must be 'km' or 'miles'")


def _generate_grid_candidates(area_df, spacing_km):
    """Return a float32 (N, 2) array of lat/lon grid points covering area_df's bounding box."""
    min_lat = float(area_df["lat"].min())
    max_lat = float(area_df["lat"].max())
    min_lon = float(area_df["lon"].min())
    max_lon = float(area_df["lon"].max())
    mid_lat = (min_lat + max_lat) / 2.0
    lat_step = spacing_km / 110.574
    lon_step = spacing_km / (111.320 * np.cos(np.radians(mid_lat)))
    lats = np.arange(min_lat, max_lat + lat_step, lat_step, dtype=np.float32)
    lons = np.arange(min_lon, max_lon + lon_step, lon_step, dtype=np.float32)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    return np.column_stack([grid_lats.ravel(), grid_lons.ravel()])


# --------------------------------------------------
# Load postcode data
# --------------------------------------------------

def build_postcode_parquet_data():
    """
    Build deployable parquet files from German postcode Excel source.

    Required input:
      raw_german_postcode_data/German Population by Postcode.xlsx
        columns: Zip/Postocde, note, Population, qkm, lat, lon

    Output:
      zipcode_parquet/postcodes_part_000.parquet  (columns: postcode, lat, lon, population)
    """
    raw_folder = DATA_FOLDER
    parquet_folder = _HERE / "zipcode_parquet"
    parquet_folder.mkdir(exist_ok=True)

    for old_file in parquet_folder.glob("*.parquet"):
        old_file.unlink()

    xlsx_file = raw_folder / "German Population by Postcode.xlsx"
    if not xlsx_file.exists():
        raise FileNotFoundError(f"Missing data file: {xlsx_file}")

    print(f"Loading {xlsx_file.name}...")
    df = pd.read_excel(xlsx_file)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Zip/Postocde": "postcode",
        "Population": "population",
    })
    df["postcode"] = df["postcode"].astype(str).str.strip().str.zfill(5)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(47, 56) & df["lon"].between(5, 16)]

    df = df[["postcode", "lat", "lon", "population"]].copy()

    chunk_size = 50_000
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start + chunk_size]
        out_file = parquet_folder / f"postcodes_part_{i:03d}.parquet"
        chunk.to_parquet(out_file, index=False)
        print(f"  Wrote {out_file.name} ({len(chunk):,} rows)")

    print(f"\nDone. {len(df):,} postcodes written to {parquet_folder}/")



#@st.cache_data(show_spinner=False)
def load_postcode_data():
    parquet_folder = _HERE / "zipcode_parquet"
    files = sorted(parquet_folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {parquet_folder}. "
            "Run build_postcode_parquet_data(2) locally first."
        )

    df = pd.concat(
        [pd.read_parquet(f, columns=_PARQUET_COLS) for f in files],
        ignore_index=True,
    )

    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(47, 56) & df["lon"].between(5, 16)]

    print(f"Loaded {len(df):,} postcodes from {len(files)} parquet files")
    return df


# --------------------------------------------------
# Filter city radius
# --------------------------------------------------

def filter_city(df, centre_lat, centre_lon, radius_km):
    distances = haversine_array(
        centre_lat,
        centre_lon,
        df["lat"].to_numpy(),
        df["lon"].to_numpy()
    )

    city_df = df.loc[distances <= radius_km].copy()

    print(f"Rows inside city radius: {len(city_df):,}")

    return city_df


# --------------------------------------------------
# Polygon / path-based area support
# --------------------------------------------------

def validate_boundary_points(boundary_points):
    if not isinstance(boundary_points, (list, tuple)):
        raise ValueError("boundary_points must be a list or tuple of (lat, lon) pairs.")

    if len(boundary_points) < 3:
        raise ValueError("At least 3 boundary points are required to define a polygon.")

    cleaned = []

    for i, pt in enumerate(boundary_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"Boundary point {i} is invalid. Each point must be (lat, lon).")

        lat, lon = pt

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise ValueError(f"Boundary point {i} contains non-numeric values.")

        if not (-90 <= lat <= 90):
            raise ValueError(f"Boundary point {i} has invalid latitude: {lat}")

        if not (-180 <= lon <= 180):
            raise ValueError(f"Boundary point {i} has invalid longitude: {lon}")

        cleaned.append((lat, lon))

    deduped = [cleaned[0]]
    for pt in cleaned[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)

    if len(deduped) < 3:
        raise ValueError("Boundary points collapse to fewer than 3 unique points.")

    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])

    if len(deduped) < 4:
        raise ValueError("Polygon must contain at least 3 unique boundary points.")

    area_proxy = polygon_area_proxy(deduped)
    if abs(area_proxy) < 1e-10:
        raise ValueError("Boundary points do not form a sensible polygon (area is effectively zero).")

    if polygon_self_intersects(deduped):
        raise ValueError("Boundary points form a self-intersecting polygon, which is not supported.")

    return deduped


def polygon_area_proxy(points):
    area = 0.0
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        area += lon1 * lat2 - lon2 * lat1
    return area / 2.0


def orientation(a, b, c):
    ay, ax = a
    by, bx = b
    cy, cx = c
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def on_segment(a, b, c):
    ay, ax = a
    by, bx = b
    cy, cx = c

    return (
        min(ax, cx) <= bx <= max(ax, cx) and
        min(ay, cy) <= by <= max(ay, cy)
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def polygon_self_intersects(points):
    n = len(points) - 1

    for i in range(n):
        p1 = points[i]
        q1 = points[i + 1]

        for j in range(i + 1, n):
            p2 = points[j]
            q2 = points[j + 1]

            if abs(i - j) <= 1:
                continue

            if i == 0 and j == n - 1:
                continue

            if segments_intersect(p1, q1, p2, q2):
                return True

    return False


def point_in_polygon(lat, lon, polygon_points):
    inside = False

    for i in range(len(polygon_points) - 1):
        lat1, lon1 = polygon_points[i]
        lat2, lon2 = polygon_points[i + 1]

        intersects = ((lat1 > lat) != (lat2 > lat))
        if intersects:
            lon_intersection = lon1 + (lon2 - lon1) * (lat - lat1) / (lat2 - lat1)
            if lon < lon_intersection:
                inside = not inside

    return inside


def points_in_polygon(test_lats, test_lons, polygon_points):
    """Vectorized ray-casting point-in-polygon test using numpy."""
    poly_lats = np.array([pt[0] for pt in polygon_points])
    poly_lons = np.array([pt[1] for pt in polygon_points])

    n_edges = len(polygon_points) - 1
    inside = np.zeros(len(test_lats), dtype=bool)

    for i in range(n_edges):
        lat1, lon1 = poly_lats[i], poly_lons[i]
        lat2, lon2 = poly_lats[i + 1], poly_lons[i + 1]

        crosses = (lat1 > test_lats) != (lat2 > test_lats)
        if not crosses.any():
            continue
        lon_intersect = lon1 + (lon2 - lon1) * (test_lats[crosses] - lat1) / (lat2 - lat1)
        inside[crosses] ^= (test_lons[crosses] < lon_intersect)

    return inside


def filter_polygon(df, boundary_points):
    polygon = validate_boundary_points(boundary_points)

    lats = np.array([pt[0] for pt in polygon])
    lons = np.array([pt[1] for pt in polygon])

    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()

    bbox_df = df[
        df["lat"].between(min_lat, max_lat) &
        df["lon"].between(min_lon, max_lon)
    ]

    print(f"Rows inside polygon bounding box: {len(bbox_df):,}")

    mask = points_in_polygon(
        bbox_df["lat"].to_numpy(),
        bbox_df["lon"].to_numpy(),
        polygon,
    )

    polygon_df = bbox_df.loc[mask].copy()

    print(f"Rows inside polygon: {len(polygon_df):,}")

    return polygon_df, polygon


# --------------------------------------------------
# Fixed-hub input validation
# --------------------------------------------------

def validate_fixed_hubs(hubs):
    if not isinstance(hubs, (list, tuple)):
        raise ValueError("hubs must be a list or tuple of (name, lat, lon) items.")

    if len(hubs) == 0:
        raise ValueError("At least one hub must be supplied.")

    cleaned = []

    for i, hub in enumerate(hubs):
        if not isinstance(hub, (list, tuple)) or len(hub) != 3:
            raise ValueError(f"Hub {i} is invalid. Each hub must be (name, lat, lon).")

        name, lat, lon = hub

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Hub {i} has an invalid name.")

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise ValueError(f"Hub {i} has non-numeric latitude/longitude.")

        if not (-90 <= lat <= 90):
            raise ValueError(f"Hub {i} has invalid latitude: {lat}")

        if not (-180 <= lon <= 180):
            raise ValueError(f"Hub {i} has invalid longitude: {lon}")

        cleaned.append((name.strip(), lat, lon))

    return cleaned


# --------------------------------------------------
# Fixed-hub coverage calculation
# --------------------------------------------------

def evaluate_fixed_hubs(df, hubs, hub_radius_km):
    if df.empty:
        raise ValueError("No postcode rows found in the selected area.")

    hubs = validate_fixed_hubs(hubs)

    demand_df = df.reset_index(drop=True)
    populations = demand_df["population"].to_numpy(dtype=np.float32)
    demand_coords_rad = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))

    tree = BallTree(demand_coords_rad, metric="haversine")
    radius_rad = hub_radius_km / EARTH_RADIUS_KM

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    hub_results = []

    for hub_num, (hub_name, hub_lat, hub_lon) in enumerate(hubs, start=1):
        hub_coord_rad = np.radians([[hub_lat, hub_lon]])
        full_cover_idx = tree.query_radius(hub_coord_rad, r=radius_rad)[0]

        new_cover_idx = full_cover_idx[~covered_mask[full_cover_idx]]

        full_population = float(populations[full_cover_idx].sum())
        new_population = float(populations[new_cover_idx].sum())
        overlap_population = full_population - new_population

        covered_mask[new_cover_idx] = True

        hub_postcode = find_nearest_postcode(hub_lat, hub_lon, demand_df)

        hub_results.append({
            "hub_number": hub_num,
            "hub_name": hub_name,
            "hub_postcode": hub_postcode,
            "lat": float(hub_lat),
            "lon": float(hub_lon),
            "postcodes": int(len(new_cover_idx)),
            "population": float(new_population),
            "potential_postcodes": int(len(full_cover_idx)),
            "potential_population": float(full_population),
            "overlap_population": float(overlap_population),
        })

        print(
            f"Evaluated hub {hub_num}: {hub_name} | "
            f"new coverage {new_population:,.0f} people | "
            f"overlap {overlap_population:,.0f}"
        )

    covered_postcodes = set(demand_df.loc[covered_mask, "postcode"].tolist())

    return hub_results, covered_postcodes


# --------------------------------------------------
# Unified map output
# --------------------------------------------------

def create_hub_map(
    hub_radius_km,
    hubs,
    unit="km",
    output_file="Hub_Map.html",
    boundary_points=None,
    centre_lat=None,
    centre_lon=None,
    city_radius_km=None,
):
    """Create an interactive Folium map showing hub placements.

    Supports both polygon-boundary and city-circle modes:
      - Polygon mode: pass boundary_points (centre is derived automatically).
      - Circle mode:  pass centre_lat, centre_lon, and city_radius_km.
    """
    if boundary_points is not None:
        polygon = validate_boundary_points(boundary_points)
        lats = [pt[0] for pt in polygon]
        lons = [pt[1] for pt in polygon]
        centre_lat = sum(lats) / len(lats)
        centre_lon = sum(lons) / len(lons)
    else:
        polygon = None

    # NOTE: do not switch this back to the default "OpenStreetMap" tiles — that
    # points at tile.openstreetmap.org, which is a light-use/dev-only demo
    # service. Its usage policy (operations.osmfoundation.org/policies/tiles)
    # explicitly says it isn't for production apps with real user traffic, and
    # that's what got this app blocked previously. Esri's WorldStreetMap tiles
    # are free, need no API key (unlike CartoDB's basemaps, which now do), and
    # are meant for embedding in apps like this one.
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=11,
        control_scale=True,
        tiles="Esri WorldStreetMap",
    )

    if unit.lower() in ["mile", "miles", "mi"]:
        hub_radius_display = hub_radius_km / KM_PER_MILE
    else:
        hub_radius_display = hub_radius_km

    if polygon is not None:
        folium.Polygon(
            locations=polygon,
            color="black",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            popup="Boundary polygon",
        ).add_to(m)
    elif city_radius_km is not None:
        city_radius_display = (
            city_radius_km / KM_PER_MILE
            if unit.lower() in ["mile", "miles", "mi"]
            else city_radius_km
        )
        folium.Marker([centre_lat, centre_lon], popup="City Centre").add_to(m)
        folium.Circle(
            [centre_lat, centre_lon],
            radius=city_radius_km * 1000,
            color="black",
            fill=False,
            popup=f"City Radius: {city_radius_display:.1f} {unit}",
        ).add_to(m)

    for i, hub in enumerate(hubs):
        color = HUB_COLORS[i % len(HUB_COLORS)]
        hub_label = hub.get("hub_name", f"Hub {hub['hub_number']}")

        popup_parts = [
            f"<b>{hub_label}</b>",
            f"Hub #{hub['hub_number']}",
            f"Postcode: {hub.get('hub_postcode', '')}",
            f"Lat/Lon: {float(hub['lat']):.6f}, {float(hub['lon']):.6f}",
            f"Postcodes: {int(hub['postcodes']):,}",
            f"Population: {int(hub['population']):,}",
        ]
        if "potential_population" in hub:
            popup_parts.append(f"Potential population: {int(hub['potential_population']):,}")
        if "overlap_population" in hub:
            popup_parts.append(f"Overlap population: {int(hub['overlap_population']):,}")

        folium.Marker(
            [float(hub["lat"]), float(hub["lon"])],
            popup="<br>".join(popup_parts),
            tooltip=hub_label,
            icon=folium.Icon(color=color),
        ).add_to(m)

        folium.Circle(
            [float(hub["lat"]), float(hub["lon"])],
            radius=hub_radius_km * 1000,
            color=color,
            fill=True,
            fill_opacity=0.18,
            popup=f"{hub_label} radius: {hub_radius_display:.1f} {unit}",
        ).add_to(m)

    m.save(output_file)
    print(f"\nMap saved to {output_file}")


# --------------------------------------------------
# Shared result printing
# --------------------------------------------------

def print_hub_results(hubs, covered_population, total_population, coverage_pct, title="HUB RESULTS"):
    print(f"\n================ {title} ================\n")

    for hub in hubs:
        label = hub.get("hub_name", f"Hub {hub['hub_number']}")
        print(label)
        print("-" * 60)
        print(f"Location:              {float(hub['lat']):.6f}, {float(hub['lon']):.6f}")
        print(f"Hub Postcode:          {hub.get('hub_postcode', '')}")
        print(f"Postcodes:             {int(hub['postcodes']):,}")
        print(f"Population:            {int(hub['population']):,}")

        if "potential_population" in hub:
            print(f"Potential Population:  {int(hub['potential_population']):,}")
        if "overlap_population" in hub:
            print(f"Overlap Population:    {int(hub['overlap_population']):,}")
            pot = hub.get("potential_population", hub["population"])
            overlap_pct = 100.0 * hub["overlap_population"] / pot if pot > 0 else 0.0
            print(f"Overlap %:             {overlap_pct:.2f}%")

        print()

    print("OVERALL COVERAGE")
    print(f"Covered population: {covered_population:,.0f} / {total_population:,.0f}")
    print(f"Coverage: {coverage_pct:.2f}%")


# --------------------------------------------------
# Fixed hubs runner: polygon mode
# --------------------------------------------------

def run_fixed_hub_coverage_polygon(
    boundary_points,
    hubs,
    hub_radius,
    radius_unit="km",
    create_map_output=True,
    map_filename="Fixed_Hub_Map_Polygon.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()

    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points define a sensible area."
        )

    hub_results, covered = evaluate_fixed_hubs(
        area_df,
        hubs,
        hub_radius_km
    )

    total_population = float(area_df["population"].sum())
    covered_population = float(
        area_df.loc[area_df["postcode"].isin(covered), "population"].sum()
    )
    coverage_pct = 0.0 if total_population == 0 else 100.0 * covered_population / total_population

    print_hub_results(hub_results, covered_population, total_population, coverage_pct, "FIXED HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hub_results,
            unit=radius_unit,
            output_file=map_filename,
            boundary_points=cleaned_polygon,
        )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, hub_results, hub_radius_km, radius_unit
    )

    return {
        "hubs": hub_results,
        "covered_postcodes": covered,
        "total_population": total_population,
        "covered_population": covered_population,
        "coverage_pct": coverage_pct,
        "boundary_points": cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


# --------------------------------------------------
# Hybrid optimisation runner
# --------------------------------------------------

def run_hybrid_optimisation_polygon(
    boundary_points,
    fixed_hubs,
    num_free_hubs,
    hub_radius,
    radius_unit="km",
    grid_spacing_km=1.0,
    map_filename="Hybrid_Hub_Map_Polygon.html",
):
    """
    Evaluate a set of user-supplied fixed hubs, then greedily optimise
    `num_free_hubs` additional locations on whatever demand remains uncovered.
    """
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()
    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points define a sensible area."
        )

    # --- Stage 1: evaluate fixed hubs ---
    fixed_results, covered_postcodes = evaluate_fixed_hubs(
        area_df, fixed_hubs, hub_radius_km
    )

    print(f"\nFixed hubs evaluated. Covered postcodes so far: {len(covered_postcodes):,}")

    # --- Stage 2: optimise free hubs on remaining demand ---
    free_results = []

    if num_free_hubs > 0:
        remaining_df = area_df[~area_df["postcode"].isin(covered_postcodes)].copy()

        if remaining_df.empty:
            print("All demand already covered by fixed hubs; no free hubs placed.")
        else:
            print(f"\nOptimising {num_free_hubs} free hub(s) on "
                  f"{len(remaining_df):,} remaining demand rows…")

            raw_free, free_covered = optimise_hubs_fast_refined(
                remaining_df,
                num_free_hubs,
                hub_radius_km,
                grid_spacing_km=grid_spacing_km,
                jostle_radius_km=2.0,
                refine_passes=3,
            )

            offset = len(fixed_results)
            for i, h in enumerate(raw_free):
                h["hub_number"] = offset + i + 1
                h["hub_name"]   = f"Optimized Hub {i + 1}"
                h.setdefault("potential_population", h["population"])
                h.setdefault("overlap_population",   0.0)

            free_results = raw_free
            covered_postcodes.update(free_covered)

    all_hubs = fixed_results + free_results

    total_population = float(area_df["population"].sum())
    covered_population = float(
        area_df.loc[area_df["postcode"].isin(covered_postcodes), "population"].sum()
    )
    coverage_pct = (
        0.0 if total_population == 0
        else 100.0 * covered_population / total_population
    )

    print_hub_results(all_hubs, covered_population, total_population, coverage_pct, "HYBRID HUB RESULTS")

    create_hub_map(
        hub_radius_km=hub_radius_km,
        hubs=all_hubs,
        unit=radius_unit,
        output_file=map_filename,
        boundary_points=cleaned_polygon,
    )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, all_hubs, hub_radius_km, radius_unit
    )

    return {
        "hubs":               all_hubs,
        "covered_postcodes":  covered_postcodes,
        "total_population":   total_population,
        "covered_population": covered_population,
        "coverage_pct":       coverage_pct,
        "boundary_points":    cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


# --------------------------------------------------
# OLD BRUTE FORCE METHOD
# --------------------------------------------------

def optimise_hubs_bruteforce(df, num_hubs, hub_radius_km):

    remaining = df.copy()
    hubs = []
    covered_postcodes = set()

    for i in range(num_hubs):

        best_score = -1
        best_location = None
        best_cover = None

        print(f"Selecting hub {i+1} (brute force)...")

        for _, candidate in remaining.iterrows():

            distances = remaining.apply(
                lambda r: haversine(candidate["lat"], candidate["lon"], r["lat"], r["lon"]),
                axis=1
            )

            covered = remaining.loc[distances <= hub_radius_km]
            score = covered["population"].sum()

            if score > best_score:
                best_score = score
                best_location = candidate
                best_cover = covered

        if best_location is None:
            break

        hub_postcode = find_nearest_postcode(best_location["lat"], best_location["lon"], df)

        hubs.append({
            "hub_number": i + 1,
            "hub_postcode": hub_postcode,
            "lat": best_location["lat"],
            "lon": best_location["lon"],
            "postcodes": len(best_cover),
            "population": best_cover["population"].sum(),
        })

        covered_postcodes.update(best_cover["postcode"])
        remaining = remaining[~remaining["postcode"].isin(best_cover["postcode"])]

        print(f"Placed hub {i+1}: {best_cover['population'].sum():,.0f} population")

    return hubs, covered_postcodes


def run_fixed_hub_coverage(
    centre_lat,
    centre_lon,
    hubs,
    hub_radius,
    city_radius,
    radius_unit="km",
    create_map_output=True,
    map_filename="Fixed_Hub_Map.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)
    city_radius_km = convert_to_km(city_radius, radius_unit)

    df = load_postcode_data()

    city_df = filter_city(df, centre_lat, centre_lon, city_radius_km)

    if city_df.empty:
        raise ValueError(
            "No postcode data found inside the city radius. "
            "Check the centre point and radius."
        )

    hub_results, covered = evaluate_fixed_hubs(
        city_df,
        hubs,
        hub_radius_km
    )

    total_population = float(city_df["population"].sum())
    covered_population = float(
        city_df.loc[city_df["postcode"].isin(covered), "population"].sum()
    )
    coverage_pct = 0.0 if total_population == 0 else 100.0 * covered_population / total_population

    print_hub_results(hub_results, covered_population, total_population, coverage_pct, "FIXED HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hub_results,
            unit=radius_unit,
            output_file=map_filename,
            centre_lat=centre_lat,
            centre_lon=centre_lon,
            city_radius_km=city_radius_km,
        )

    return {
        "hubs": hub_results,
        "covered_postcodes": covered,
        "total_population": total_population,
        "covered_population": covered_population,
        "coverage_pct": coverage_pct
    }


# --------------------------------------------------
# FAST OPTIMIZED METHOD
# --------------------------------------------------

def optimise_hubs_fast(df, num_hubs, hub_radius_km, candidate_stride=1):

    demand_df = df.reset_index(drop=True)

    if candidate_stride > 1:
        candidate_df = demand_df.iloc[::candidate_stride].reset_index(drop=True)
    else:
        candidate_df = demand_df.copy()

    demand_coords = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))
    candidate_coords = np.radians(candidate_df[["lat", "lon"]].to_numpy(dtype=np.float32))

    tree = BallTree(demand_coords, metric="haversine")
    radius_rad = hub_radius_km / EARTH_RADIUS_KM

    print("Precomputing coverage...")
    neighbor_indices = tree.query_radius(candidate_coords, r=radius_rad)

    populations = demand_df["population"].to_numpy(dtype=np.float32)

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    hubs = []

    for hub_num in range(1, num_hubs + 1):

        best_idx = None
        best_gain = -1
        best_cover = None

        print(f"Selecting hub {hub_num} (optimized)...")

        for idx, cover in enumerate(neighbor_indices):

            uncovered = cover[~covered_mask[cover]]
            gain = populations[uncovered].sum()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_cover = uncovered

        if best_idx is None:
            break

        covered_mask[best_cover] = True

        hub_row = candidate_df.iloc[best_idx]

        hub_postcode = find_nearest_postcode(hub_row["lat"], hub_row["lon"], demand_df)

        hubs.append({
            "hub_number": hub_num,
            "hub_postcode": hub_postcode,
            "lat": hub_row["lat"],
            "lon": hub_row["lon"],
            "postcodes": len(best_cover),
            "population": populations[best_cover].sum(),
        })

        print(f"Placed hub {hub_num}: {populations[best_cover].sum():,.0f} population")

    covered_postcodes = set(demand_df.loc[covered_mask, "postcode"])

    return hubs, covered_postcodes


def optimise_hubs_fast_refined(
    df,
    num_hubs,
    hub_radius_km,
    grid_spacing_km=1.0,
    jostle_radius_km=2.0,
    refine_passes=3,
    min_improvement_population=1.0
):
    if df.empty:
        raise ValueError("No postcode rows found inside the search area.")

    demand_df = df.reset_index(drop=True)

    candidate_latlons = _generate_grid_candidates(demand_df, grid_spacing_km)

    demand_coords = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))
    candidate_coords = np.radians(candidate_latlons)

    demand_tree = BallTree(demand_coords, metric="haversine")
    candidate_tree = BallTree(candidate_coords, metric="haversine")

    hub_radius_rad = hub_radius_km / EARTH_RADIUS_KM
    jostle_radius_rad = jostle_radius_km / EARTH_RADIUS_KM

    print(f"Precomputing hub coverage for {len(candidate_latlons):,} grid candidates ({grid_spacing_km} km spacing)...")
    neighbor_indices = demand_tree.query_radius(candidate_coords, r=hub_radius_rad)

    populations = demand_df["population"].to_numpy(dtype=np.float32)

    # --- Stage 1: Greedy seed solution ---

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    selected_candidate_indices = []

    for hub_num in range(1, num_hubs + 1):
        best_idx = None
        best_gain = -1.0
        best_cover = None

        print(f"Selecting hub {hub_num} (greedy seed)...")

        for idx, cover in enumerate(neighbor_indices):
            uncovered = cover[~covered_mask[cover]]
            gain = populations[uncovered].sum()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_cover = uncovered

        if best_idx is None or best_cover is None or len(best_cover) == 0:
            print(f"No further useful hub placement found after {hub_num - 1} hub(s).")
            break

        selected_candidate_indices.append(best_idx)
        covered_mask[best_cover] = True

        print(f"Placed seed hub {hub_num}: {populations[best_cover].sum():,.0f} population")

    if not selected_candidate_indices:
        return [], set()

    # --- Helper: summarize chosen hubs ---

    def summarize_selection(selected_indices):
        overall_mask = np.zeros(len(demand_df), dtype=bool)
        hubs = []

        for hub_num, candidate_idx in enumerate(selected_indices, start=1):
            hub_lat = float(candidate_latlons[candidate_idx][0])
            hub_lon = float(candidate_latlons[candidate_idx][1])
            full_cover = neighbor_indices[candidate_idx]
            hub_postcode = find_nearest_postcode(hub_lat, hub_lon, demand_df)

            other_indices = [idx for i, idx in enumerate(selected_indices) if (i + 1) != hub_num]
            others_mask = np.zeros(len(demand_df), dtype=bool)
            for other_idx in other_indices:
                others_mask[neighbor_indices[other_idx]] = True

            net_new_cover = full_cover[~others_mask[full_cover]]

            potential_population = float(populations[full_cover].sum())
            net_population = float(populations[net_new_cover].sum())
            overlap_population = potential_population - net_population

            overall_mask[full_cover] = True

            hubs.append({
                "hub_number": hub_num,
                "hub_postcode": hub_postcode,
                "lat": hub_lat,
                "lon": hub_lon,
                "postcodes": int(len(net_new_cover)),
                "population": float(net_population),
                "potential_postcodes": int(len(full_cover)),
                "potential_population": float(potential_population),
                "overlap_population": float(overlap_population),
            })

        covered_postcodes = set(demand_df.loc[overall_mask, "postcode"])
        return hubs, covered_postcodes

    # --- Helper: total unique covered population ---

    def total_unique_population(selected_indices):
        mask = np.zeros(len(demand_df), dtype=bool)
        for idx in selected_indices:
            mask[neighbor_indices[idx]] = True
        return float(populations[mask].sum())

    # --- Stage 2: Local refinement / jostling ---

    current_total = total_unique_population(selected_candidate_indices)
    print(f"\nInitial greedy unique covered population: {current_total:,.0f}")

    for refine_pass in range(1, refine_passes + 1):
        improved_this_pass = False
        print(f"\nRefinement pass {refine_pass}/{refine_passes}...")

        for hub_pos in range(len(selected_candidate_indices)):
            current_idx = selected_candidate_indices[hub_pos]

            others_mask = np.zeros(len(demand_df), dtype=bool)
            for j, idx in enumerate(selected_candidate_indices):
                if j != hub_pos:
                    others_mask[neighbor_indices[idx]] = True

            current_net_cover = neighbor_indices[current_idx][~others_mask[neighbor_indices[current_idx]]]
            current_net_gain = float(populations[current_net_cover].sum())

            nearby_candidate_indices = candidate_tree.query_radius(
                candidate_coords[current_idx:current_idx + 1],
                r=jostle_radius_rad
            )[0]

            best_local_idx = current_idx
            best_local_gain = current_net_gain

            for candidate_idx in nearby_candidate_indices:
                candidate_net_cover = neighbor_indices[candidate_idx][~others_mask[neighbor_indices[candidate_idx]]]
                candidate_net_gain = float(populations[candidate_net_cover].sum())

                if candidate_net_gain > best_local_gain + min_improvement_population:
                    best_local_gain = candidate_net_gain
                    best_local_idx = candidate_idx

            if best_local_idx != current_idx:
                trial_selection = selected_candidate_indices.copy()
                trial_selection[hub_pos] = best_local_idx

                trial_total = total_unique_population(trial_selection)

                if trial_total > current_total + min_improvement_population:
                    print(
                        f"Hub {hub_pos + 1} moved: "
                        f"{current_total:,.0f} -> {trial_total:,.0f} "
                        f"(+{trial_total - current_total:,.0f})"
                    )
                    selected_candidate_indices = trial_selection
                    current_total = trial_total
                    improved_this_pass = True

        if not improved_this_pass:
            print("No improvements found in this pass.")
            break

    print(f"\nFinal refined unique covered population: {current_total:,.0f}")

    hubs, covered_postcodes = summarize_selection(selected_candidate_indices)
    return hubs, covered_postcodes


# --------------------------------------------------
# RUNNER
# --------------------------------------------------

def run_hub_optimisation(
    centre_lat,
    centre_lon,
    num_hubs,
    hub_radius,
    city_radius,
    radius_unit="km",
    use_optimized=True,
    grid_spacing_km=1.0,
    create_map_output=True
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)
    city_radius_km = convert_to_km(city_radius, radius_unit)

    df = load_postcode_data()

    city_df = filter_city(df, centre_lat, centre_lon, city_radius_km)

    if use_optimized:
        hubs, covered = optimise_hubs_fast_refined(
            city_df,
            num_hubs,
            hub_radius_km,
            grid_spacing_km=grid_spacing_km,
            jostle_radius_km=3.0,
            refine_passes=5
        )
    else:
        hubs, covered = optimise_hubs_bruteforce(
            city_df,
            num_hubs,
            hub_radius_km
        )

    total_population = city_df["population"].sum()
    covered_population = city_df.loc[
        city_df["postcode"].isin(covered),
        "population"
    ].sum()

    coverage_pct = 100 * covered_population / total_population

    print_hub_results(hubs, covered_population, total_population, coverage_pct)

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hubs,
            unit=radius_unit,
            centre_lat=centre_lat,
            centre_lon=centre_lon,
            city_radius_km=city_radius_km,
        )


def run_hub_optimisation_polygon(
    boundary_points,
    num_hubs,
    hub_radius,
    radius_unit="km",
    use_optimized=True,
    grid_spacing_km=1.0,
    create_map_output=True,
    map_filename="Hub_Map_Polygon.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()

    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points are in the right order and cover a sensible area."
        )

    if use_optimized:
        hubs, covered = optimise_hubs_fast_refined(
            area_df,
            num_hubs,
            hub_radius_km,
            grid_spacing_km=grid_spacing_km,
            jostle_radius_km=2.0,
            refine_passes=3
        )
    else:
        hubs, covered = optimise_hubs_bruteforce(
            area_df,
            num_hubs,
            hub_radius_km
        )

    total_population = area_df["population"].sum()
    covered_population = area_df.loc[
        area_df["postcode"].isin(covered),
        "population"
    ].sum()

    coverage_pct = 0 if total_population == 0 else 100 * covered_population / total_population

    print_hub_results(hubs, covered_population, total_population, coverage_pct, "POLYGON HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hubs,
            unit=radius_unit,
            output_file=map_filename,
            boundary_points=cleaned_polygon,
        )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, hubs, hub_radius_km, radius_unit
    )

    return {
        "hubs": hubs,
        "covered_postcodes": covered,
        "total_population": float(total_population),
        "covered_population": float(covered_population),
        "coverage_pct": float(coverage_pct),
        "boundary_points": cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


def find_nearest_postcode(lat, lon, df):
    distances = haversine_array(
        lat,
        lon,
        df["lat"].to_numpy(),
        df["lon"].to_numpy()
    )

    nearest_idx = int(np.argmin(distances))
    return df.iloc[nearest_idx]["postcode"]


def build_postcode_hub_mappings(area_df, hub_results, hub_radius_km, radius_unit="miles"):
    """
    Returns two DataFrames:
      multi_df  – one row per (postcode, hub) pair where the hub covers that postcode.
      single_df – one row per postcode, assigned to its nearest covering hub
                  (tie-broken by hub_number, i.e. order placed during optimisation).
    """
    dist_col = f"Distance ({radius_unit})"
    cols = ["Postcode", "Lat", "Lon", "Hub", "Hub Number", dist_col]

    if area_df.empty or not hub_results:
        empty = pd.DataFrame(columns=cols)
        return empty, empty.copy()

    pc_lats = area_df["lat"].to_numpy(dtype=np.float32)
    pc_lons = area_df["lon"].to_numpy(dtype=np.float32)
    use_miles = radius_unit.lower() in ("mile", "miles", "mi")

    multi_dfs = []
    # df_idx -> (dist_km, hub_number, hub_name, dist_display)
    best: dict[int, tuple] = {}

    for hub in hub_results:
        hub_lat    = hub["lat"]
        hub_lon    = hub["lon"]
        hub_name   = hub.get("hub_name", f"Hub {hub['hub_number']}")
        hub_number = hub["hub_number"]

        dists_km = haversine_array(hub_lat, hub_lon, pc_lats, pc_lons)
        in_range = np.where(dists_km <= hub_radius_km)[0]

        if len(in_range) == 0:
            continue

        dists_display = dists_km[in_range] / KM_PER_MILE if use_miles else dists_km[in_range]

        batch = area_df.iloc[in_range][["postcode", "lat", "lon"]].copy()
        batch.columns = ["Postcode", "Lat", "Lon"]
        batch["Lat"] = batch["Lat"].round(5)
        batch["Lon"] = batch["Lon"].round(5)
        batch["Hub"] = hub_name
        batch["Hub Number"] = hub_number
        batch[dist_col] = np.round(dists_display, 4)
        multi_dfs.append(batch[cols])

        # nearest-hub tracking
        for arr_pos, df_idx in enumerate(in_range):
            d_km  = float(dists_km[df_idx])
            d_dis = float(dists_display[arr_pos])
            prev  = best.get(df_idx)
            if prev is None or d_km < prev[0] or (d_km == prev[0] and hub_number < prev[1]):
                best[df_idx] = (d_km, hub_number, hub_name, d_dis)

    multi_df = (
        pd.concat(multi_dfs, ignore_index=True)
        .sort_values(["Hub Number", "Postcode"])
        .reset_index(drop=True)
        if multi_dfs else pd.DataFrame(columns=cols)
    )

    if best:
        best_indices = list(best.keys())
        best_vals    = list(best.values())
        single_df = area_df.iloc[best_indices][["postcode", "lat", "lon"]].copy()
        single_df.columns = ["Postcode", "Lat", "Lon"]
        single_df["Lat"]        = single_df["Lat"].round(5)
        single_df["Lon"]        = single_df["Lon"].round(5)
        single_df["Hub Number"] = [v[1] for v in best_vals]
        single_df["Hub"]        = [v[2] for v in best_vals]
        single_df[dist_col]     = [round(v[3], 4) for v in best_vals]
        single_df = (
            single_df[cols]
            .sort_values(["Hub Number", "Postcode"])
            .reset_index(drop=True)
        )
    else:
        single_df = pd.DataFrame(columns=cols)

    return multi_df, single_df



if __name__ == "__main__":
    # Uncomment to (re)build parquet files from raw Excel data:
    # build_postcode_parquet_data()

    berlin_boundary = [
        (52.6755, 13.0884),
        (52.6755, 13.7612),
        (52.3382, 13.7612),
        (52.3382, 13.0884),
        (52.6755, 13.0884),
    ]

    run_hub_optimisation_polygon(
        boundary_points=berlin_boundary,
        num_hubs=4,
        hub_radius=5,
        radius_unit="km",
        use_optimized=True,
        grid_spacing_km=1.0,
        create_map_output=True,
        map_filename="Berlin-4_hubs-5km.html"
    )