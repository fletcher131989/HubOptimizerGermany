import base64
import json

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from main import (
    run_hub_optimisation_polygon,
    run_fixed_hub_coverage_polygon,
    run_hybrid_optimisation_polygon,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def download_buttons(multi_df, single_df, key_prefix="top"):
    """Render the two postcode download buttons side-by-side."""
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="⬇️ Download covered postcodes (all hubs)",
            data=multi_df.to_csv(index=False),
            file_name="covered_postcodes_all_hubs.csv",
            mime="text/csv",
            key=f"{key_prefix}_multi",
            help="Each postcode appears once per hub that covers it.",
            width='stretch',
        )

    with c2:
        st.download_button(
            label="⬇️ Download covered postcodes (nearest hub only)",
            data=single_df.to_csv(index=False),
            file_name="covered_postcodes_nearest_hub.csv",
            mime="text/csv",
            key=f"{key_prefix}_single",
            help="Each postcode is assigned exclusively to its nearest hub.",
            width='stretch',
        )


def geojson_polygon_to_latlon_list(geojson_geometry):
    if not geojson_geometry:
        raise ValueError("No geometry supplied.")
    if geojson_geometry["type"] != "Polygon":
        raise ValueError("Only Polygon geometries are supported.")
    ring = geojson_geometry["coordinates"][0]
    return [(lat, lon) for lon, lat in ring]


def show_overlay(placeholder, message="Optimizing…", subtext="This may take a moment."):
    placeholder.markdown(f"""
    <style>
    .hub-overlay {{
        position: fixed;
        inset: 0;
        background: rgba(14, 17, 23, 0.78);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hub-overlay-box {{
        background: #1e2130;
        border: 1px solid #3a3f5c;
        border-radius: 16px;
        padding: 2.5rem 3.5rem;
        text-align: center;
        color: #f0f2f6;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        min-width: 280px;
    }}
    .hub-spinner {{
        width: 52px;
        height: 52px;
        border: 5px solid #3a3f5c;
        border-top-color: #e05c5c;
        border-radius: 50%;
        animation: hub-spin 0.85s linear infinite;
        margin: 0 auto 1.4rem;
    }}
    @keyframes hub-spin {{
        to {{ transform: rotate(360deg); }}
    }}
    .hub-overlay-box h3 {{
        margin: 0 0 0.4rem;
        font-size: 1.25rem;
        font-weight: 600;
    }}
    .hub-overlay-box p {{
        margin: 0;
        opacity: 0.65;
        font-size: 0.9rem;
    }}
    </style>
    <div class="hub-overlay">
        <div class="hub-overlay-box">
            <div class="hub-spinner"></div>
            <h3>{message}</h3>
            <p>{subtext}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_results_df(hubs):
    rows = []
    for h in hubs:
        pot_pop = h.get("potential_population") or h["population"]
        overlap = h.get("overlap_population", 0.0)
        overlap_pct = (100.0 * overlap / pot_pop) if pot_pop > 0 else 0.0
        rows.append({
            "Hub":            h.get("hub_name", f"Hub {h['hub_number']}"),
            "Postcode":       h.get("hub_postcode", ""),
            "Latitude":       round(h["lat"], 5),
            "Longitude":      round(h["lon"], 5),
            "New Postcodes":  int(h["postcodes"]),
            "New Population": int(h["population"]),
            "Potential Pop.": int(pot_pop),
            "Overlap Pop.":   int(overlap),
            "Overlap %":      f"{overlap_pct:.1f}%",
        })
    return pd.DataFrame(rows)


def render_results(result, map_filename="user_polygon_result.html"):
    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Population (area)", f"{result['total_population']:,.0f}")
    c2.metric("Covered Population",      f"{result['covered_population']:,.0f}")
    c3.metric("Coverage",                f"{result['coverage_pct']:.1f}%")

    # ← NEW: top-level download buttons
    multi_df  = result.get("multi_hub_df",  pd.DataFrame())
    single_df = result.get("single_hub_df", pd.DataFrame())
    if not multi_df.empty:
        st.markdown("#### Download Covered Postcodes")
        download_buttons(multi_df, single_df, key_prefix="top")

    st.markdown("#### Hub Summary")
    st.dataframe(build_results_df(result["hubs"]), width='stretch', hide_index=True)

    if not multi_df.empty:
        st.markdown("#### Per-Hub Downloads")
        for h in result["hubs"]:
            hub_label = h.get("hub_name", f"Hub {h['hub_number']}")
            postcode  = h.get("hub_postcode", "")
            hub_num   = h["hub_number"]

            with st.expander(f"{hub_label}  —  {postcode}"):
                hub_multi  = multi_df[multi_df["Hub Number"] == hub_num]
                hub_single = single_df[single_df["Hub Number"] == hub_num]
                download_buttons(hub_multi, hub_single, key_prefix=f"hub_{hub_num}")

    st.caption("Map output saved to `user_polygon_result.html`.")
    st.markdown("#### Coverage Map")
    try:
        with open(map_filename, "r", encoding="utf-8") as f:
            map_html = f.read()
        encoded = base64.b64encode(map_html.encode()).decode()
        st.iframe(f"data:text/html;base64,{encoded}", height=550)
    except FileNotFoundError:
        st.warning("Map file not found — it may not have been generated.")


# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(page_title="Germany Hub Optimizer", layout="wide")
st.title("🇩🇪 Germany Hub Optimizer")

# --------------------------------------------------
# Sidebar — optimization settings
# --------------------------------------------------

st.sidebar.header("Optimization Settings")
num_hubs = st.sidebar.number_input("Total number of hubs", min_value=1, max_value=50, value=4)
hub_radius = st.sidebar.number_input("Hub radius", min_value=0.1, value=5.0)
radius_unit = st.sidebar.selectbox("Radius unit", ["miles", "km"], index=0)
grid_spacing_km = st.sidebar.number_input(
    "Grid spacing (km)",
    min_value=0.25,
    max_value=5.0,
    value=1.0,
    step=0.25,
    help="Distance between candidate hub locations. Larger = faster and less memory. 1 km is a good default.",
)

# --------------------------------------------------
# Sidebar — fixed hubs
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("Fixed Hubs (optional)")

max_fixed = int(num_hubs)
num_fixed = int(st.sidebar.number_input(
    "Number of fixed hubs",
    min_value=0,
    max_value=max_fixed,
    value=0,
    help=f"Pin up to {max_fixed} location(s). Remaining hubs will be optimized automatically.",
))

fixed_hubs_input = []
for i in range(num_fixed):
    st.sidebar.markdown(f"**Fixed Hub {i + 1}**")
    name = st.sidebar.text_input("Name", key=f"fh_name_{i}", value=f"Fixed Hub {i + 1}")
    col_a, col_b = st.sidebar.columns(2)
    lat = col_a.number_input("Lat", key=f"fh_lat_{i}", value=52.5200, format="%.5f", step=0.001)
    lon = col_b.number_input("Lon", key=f"fh_lon_{i}", value=13.4050, format="%.5f", step=0.001)
    fixed_hubs_input.append((name.strip(), float(lat), float(lon)))

if num_fixed > 0:
    num_free = int(num_hubs) - num_fixed
    if num_free > 0:
        st.sidebar.caption(f"↳ {num_free} hub(s) will be optimized automatically.")
    else:
        st.sidebar.caption("↳ All hubs are fixed — no optimization will run.")

# --------------------------------------------------
# Map
# --------------------------------------------------

base_map = folium.Map(location=[51.1, 10.4], zoom_start=6, control_scale=True)
Draw(
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True,
    },
    edit_options={"edit": True, "remove": True},
).add_to(base_map)

st.write("Draw a polygon or rectangle on the map to define your search area.")
map_data = st_folium(base_map, width=1000, height=600, key=f"map_{st.session_state.get('map_key', 0)}")

# --------------------------------------------------
# Extract geometry — no st.stop() used here
# --------------------------------------------------

raw_drawings = map_data.get("all_drawings") or []
drawn_features = [f for f in raw_drawings if f is not None]
geometry = drawn_features[-1].get("geometry") if drawn_features else None

if not geometry:
    st.info("No polygon drawn yet. Use the drawing tools on the left side of the map.")

else:
    try:
        boundary_points = geojson_polygon_to_latlon_list(geometry)
    except Exception as e:
        st.error(str(e))
        boundary_points = None

    if boundary_points is not None:
        n_pts = len(boundary_points)
        st.success(f"✅ Area defined — {n_pts} boundary point{'s' if n_pts != 1 else ''}.")

        with st.expander("View boundary points"):
            preview = (
                boundary_points[:4] + [["...", "..."]] + boundary_points[-4:]
                if n_pts > 8 else boundary_points
            )
            st.code(json.dumps(preview, indent=2), language="json")

        num_free = int(num_hubs) - len(fixed_hubs_input)
        run_label = (
            f"🚀 Run Optimization  ({len(fixed_hubs_input)} fixed · {num_free} optimized)"
            if fixed_hubs_input else "🚀 Run Optimization"
        )

        if st.button(run_label, type="primary"):
            overlay = st.empty()
            show_overlay(
                overlay,
                message="Optimizing…",
                subtext=f"Placing {int(num_hubs)} hub(s) across {n_pts} boundary points.",
            )

            try:
                if fixed_hubs_input and num_free > 0:
                    result = run_hybrid_optimisation_polygon(
                        boundary_points=boundary_points,
                        fixed_hubs=fixed_hubs_input,
                        num_free_hubs=num_free,
                        hub_radius=hub_radius,
                        radius_unit=radius_unit,
                        grid_spacing_km=float(grid_spacing_km),
                        map_filename="user_polygon_result.html",
                    )
                elif fixed_hubs_input and num_free == 0:
                    result = run_fixed_hub_coverage_polygon(
                        boundary_points=boundary_points,
                        hubs=fixed_hubs_input,
                        hub_radius=hub_radius,
                        radius_unit=radius_unit,
                        create_map_output=True,
                        map_filename="user_polygon_result.html",
                    )
                else:
                    result = run_hub_optimisation_polygon(
                        boundary_points=boundary_points,
                        num_hubs=int(num_hubs),
                        hub_radius=hub_radius,
                        radius_unit=radius_unit,
                        use_optimized=True,
                        grid_spacing_km=float(grid_spacing_km),
                        create_map_output=True,
                        map_filename="user_polygon_result.html",
                    )
                overlay.empty()
                # Persist results so they survive Streamlit reruns
                st.session_state["result"] = result
                # Reset map key to force a fresh map render (clears drawn area)
                st.session_state["map_key"] = st.session_state.get("map_key", 0) + 1
                st.rerun()
            except Exception as e:
                overlay.empty()
                st.error(str(e))

if "result" in st.session_state:
    render_results(st.session_state["result"])