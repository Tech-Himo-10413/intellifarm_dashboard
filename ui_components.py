"""
ui_components.py
================
Handles visual styling and dynamic Plotly dashboard generation.

Supported chart types: bar, pie, line, scatter, histogram, treemap, funnel, map.
All builders fall back gracefully if columns are missing or data is sparse.
"""
import contextlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import json
import os
import contextlib



def scroll_to_top():
    import time
    ts = int(time.time() * 1000)
    st.components.v1.html(
        f"""
        <script>
        // TS: {ts}
        setTimeout(function() {{
            const el = window.parent.document.getElementById('top_of_page') || document.getElementById('top_of_page');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
            }} else {{
                const targets = [
                    window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                    window.parent.document.querySelector('.main'),
                    window.parent.document.documentElement,
                    window.parent.document.body,
                    document.querySelector('[data-testid="stAppViewContainer"]'),
                    document.documentElement
                ];
                targets.forEach(t => {{ if (t) {{ t.scrollTo({{top: 0, behavior: 'smooth'}}); }} }});
            }}
        }}, 150);
        </script>
        """, height=0
    )

def scroll_down_slightly():
    import time
    ts = int(time.time() * 1000)
    st.components.v1.html(
        f"""
        <script>
        // TS: {ts}
        setTimeout(function() {{
            const targets = [
                window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                window.parent.document.querySelector('.main'),
                window.parent.document.documentElement,
                window.parent.document.body,
                document.querySelector('[data-testid="stAppViewContainer"]'),
                document.documentElement
            ];
            targets.forEach(t => {{ if (t) {{ t.scrollBy({{top: 400, behavior: 'smooth'}}); }} }});
        }}, 150);
        </script>
        """, height=0
    )

# ─────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────
BRAND_BLUE = "#1F618D"
PALETTE = [
    "#27AE60", "#2E86C1", "#1ABC9C", "#2980B9", "#117A65",
    "#1ABC9C", "#3498DB", "#16A085", "#2471A3", "#52BE80"
]
FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"


# ─────────────────────────────────────────────────────────
# 1. GLOBAL CSS & GEOCODING
# ─────────────────────────────────────────────────────────

def apply_custom_css() -> None:
    """Injects dynamic CSS that automatically respects Streamlit's native Light/Dark themes."""
    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome but keep the menu for theme switching */
        #MainMenu, footer { visibility: hidden; }


        /* Comfortable main content padding */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* Metric cards - Uses the theme's primary color (Green) */
        div[data-testid="stMetricValue"] {
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--primary-color); 
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: var(--text-color);
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Tab styling - Dynamically highlights active tabs */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--primary-color) !important;
            border-bottom: 2px solid var(--primary-color) !important;
        }

        /* Progress bar color */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color);
        }

        /* Insight card - Adapts to Light or Dark mode backgrounds */
        .insight-box {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-left: 4px solid var(--primary-color);
            padding: 0.85rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        /* Instruction Box - Soft warning styling for clear directions */
        .instruction-box {
            background-color: #FFF8E1; /* Very soft yellow background */
            color: #5D4037; /* Dark brown text for high contrast */
            border-left: 5px solid #FFCA28; /* Bold yellow accent border */
            padding: 1.2rem 1.5rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .instruction-box h4 {
            margin-top: 0;
            color: #E65100 !important; /* Deep orange header */
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .instruction-box p { font-size: 0.95rem; margin-bottom: 0.5rem; font-weight: 500;}
        .instruction-box ul { margin-bottom: 0; padding-left: 1.5rem; }
        .instruction-box li { font-size: 0.95rem; margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
@st.cache_data(show_spinner=False)
def geocode_locations(locations_list: list, suffix: str = ", India") -> tuple:
    """Translates text names like 'Machamara' into exact coordinates using OpenStreetMap and a persistent cache."""
    cache_file = "geocache.json"
    coords_map = {}
    
    # Load cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                coords_map = json.load(f)
        except Exception:
            pass

    geolocator = Nominatim(user_agent="intellifarm_dashboard_agent")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) 
    
    failed_locs = []
    new_geocodes = False
    
    for loc in locations_list:
        if pd.isna(loc) or not str(loc).strip(): 
            continue
            
        loc_str = str(loc).strip()
        query = f"{loc_str}{suffix}" 
        
        # Check cache first
        if loc_str in coords_map:
            continue
            
        try:
            location_data = geocode(query)
            if location_data:
                coords_map[loc_str] = (location_data.latitude, location_data.longitude)
                new_geocodes = True
            else:
                failed_locs.append(loc_str)
        except Exception:
            failed_locs.append(loc_str)
            
    # Save cache if updated
    if new_geocodes:
        try:
            with open(cache_file, "w") as f:
                json.dump(coords_map, f)
        except Exception:
            pass
            
    return coords_map, failed_locs


# ─────────────────────────────────────────────────────────
# 2. MAIN CHART DISPATCHER
# ─────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, cfg: Dict) -> Tuple[Optional[go.Figure], Optional[str], Optional[str]]:
    """
    Builds the best-fitting Plotly chart for the given data and config.
    Returns (figure, error_message, warning_message).
    """
    if df is None or df.empty or df.columns.empty:
        return None, "Data is empty.", None

    cols = df.columns.tolist()
    title = cfg.get("title", "Data Insight")
    chart_type = cfg.get("chart_type", "bar").lower()

    # Resolve columns from config, with positional fallbacks
    x_col = _resolve_col(df, cfg.get("x_column"), cols, 0)
    y_col = _resolve_col(df, cfg.get("y_column"), cols, 1)
    color_col = cfg.get("color_column") if cfg.get("color_column") in cols else None

    # 🟢 Guarantee colors for bar charts to make them visually attractive!
    if chart_type == "bar" and not color_col:
        color_col = x_col

    # ── Special: single scalar result (e.g. SELECT COUNT(*) FROM dataset) ──
    if len(cols) == 1 and len(df) == 1:
        val = df.iloc[0, 0]
        numeric_val = float(val) if _is_numeric(val) else 0
        fig = go.Figure(
            go.Indicator(
                mode="number",
                value=numeric_val,
                title={"text": title, "font": {"size": 16}},
                number={"font": {"size": 72, "color": BRAND_BLUE}},
            )
        )
        return _apply_layout(fig, title), None, None

    # ── Auto-correct chart type based on data shape ──
    chart_type = _auto_correct_chart_type(df, x_col, y_col, chart_type)

    # ── Dispatch ──
    warning_msg = None
    try:
        if chart_type == "map":
            fig, map_warn = _map(df, cfg)
            if map_warn:
                warning_msg = map_warn
            if fig is None:
                raise ValueError("Insufficient data or geocoding failed.")
        elif chart_type == "pie":
            fig = _pie(df, x_col, y_col)
        elif chart_type == "line":
            fig = _line(df, x_col, y_col, color_col, cfg)
        elif chart_type == "scatter":
            fig = _scatter(df, x_col, y_col, color_col, cfg)
        elif chart_type == "histogram":
            fig = _histogram(df, x_col)
        elif chart_type == "treemap":
            fig = _treemap(df, x_col, y_col)
        elif chart_type == "funnel":
            fig = _funnel(df, x_col, y_col)
        else:
            fig = _bar(df, x_col, y_col, color_col, cfg)

        return _apply_layout(fig, title), None, warning_msg

    except Exception as e:
        err_msg = (
            f"❌ **Visual Rendering Error:** Could not build the requested {chart_type.title()} chart.\n\n"
            f"💡 **What to do:** The shape of the data returned by the AI might not match the strict requirements "
            f"for a {chart_type.title()}. I have automatically generated a fallback Bar Chart so you can still view the data."
        )
        # Last-resort fallback: simple horizontal bar
        try:
            fig = px.bar(df, x=y_col, y=x_col, orientation="h",
                         color_discrete_sequence=PALETTE)
            return _apply_layout(fig, title), err_msg, warning_msg
        except Exception:
            return None, err_msg, warning_msg


# ─────────────────────────────────────────────────────────
# 3. HOVER-DETAIL HELPERS
# ─────────────────────────────────────────────────────────

def _dynamic_hover_data(df: pd.DataFrame, shown_cols) -> Dict:
    """
    Builds a plotly-express `hover_data` dict that includes every column
    in the result set NOT already shown elsewhere.
    """
    hover_data: Dict = {}
    for col in df.columns:
        if col in shown_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            hover_data[col] = ":,.2f"
        else:
            hover_data[col] = True
    return hover_data


def _hover_kwargs(df: pd.DataFrame, primary_col: str, *other_shown_cols: str) -> Dict:
    """
    Convenience wrapper: returns {"hover_name": ..., "hover_data": ...}
    ready to splice into a plotly.express call.
    """
    shown = {primary_col, *other_shown_cols}
    hover_data = _dynamic_hover_data(df, shown)
    hover_data[primary_col] = False  # avoid duplicating the bold hover_name
    return {"hover_name": primary_col, "hover_data": hover_data}


# ─────────────────────────────────────────────────────────
# 4. INDIVIDUAL CHART BUILDERS
# ─────────────────────────────────────────────────────────

def _map(df: pd.DataFrame, cfg: Dict) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Builds a geospatial map, handling both coordinates and text locations with graceful degradation.
    Returns (Figure, warning_message)."""
    lat_col = cfg.get("lat_column")
    lon_col = cfg.get("lon_column")
    loc_col = cfg.get("location_column")
    color_col = cfg.get("color_column") if cfg.get("color_column") in df.columns else None
    size_col = cfg.get("size_column") if cfg.get("size_column") in df.columns else None

    # ── Scenario A: Exact Coordinates are provided ──
    if lat_col in df.columns and lon_col in df.columns:
        # Dynamic zoom and center
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        lat_range = df[lat_col].max() - df[lat_col].min()
        lon_range = df[lon_col].max() - df[lon_col].min()
        max_range = max(lat_range, lon_range)
        
        if max_range == 0: zoom = 6
        elif max_range > 15: zoom = 4
        elif max_range > 8: zoom = 5
        elif max_range > 4: zoom = 6
        elif max_range > 2: zoom = 7
        elif max_range > 1: zoom = 8
        elif max_range > 0.5: zoom = 9
        elif max_range > 0.2: zoom = 10
        elif max_range > 0.1: zoom = 11
        elif max_range > 0.05: zoom = 12
        else: zoom = 13

        hover_kwargs = _hover_kwargs(df, loc_col or lat_col, lat_col, lon_col)
        fig = px.scatter_mapbox(
            df, lat=lat_col, lon=lon_col, color=color_col, size=size_col,
            color_discrete_sequence=PALETTE, mapbox_style="white-bg", 
            zoom=zoom, center={"lat": center_lat, "lon": center_lon},
            **hover_kwargs
        )
        fig.update_layout(
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google Maps Satellite",
                    "source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"]
                }
            ],
            margin={"r":0,"t":0,"l":0,"b":0},
            hovermode="closest"
        )
        return fig, None

    if loc_col in df.columns:
        unique_locs = df[loc_col].dropna().unique().tolist()
        
        coords_map, failed = geocode_locations(unique_locs)
        
        warning_msg = None
        if failed:
            warning_msg = (
                f"⚠️ **Mapping Notice:** Could not find exact map coordinates for {len(failed)} location(s) "
                f"(e.g., {', '.join(map(str, failed[:3]))}). They are excluded from this map.\n\n"
                "💡 **What to do:** Ensure village/city names are spelled correctly in your dataset."
            )
        
        plot_df = df.copy()
        plot_df['__lat__'] = plot_df[loc_col].map(lambda x: coords_map.get(x, (None, None))[0])
        plot_df['__lon__'] = plot_df[loc_col].map(lambda x: coords_map.get(x, (None, None))[1])
        plot_df = plot_df.dropna(subset=['__lat__', '__lon__'])
        
        # Actionable Error for total failure
        if plot_df.empty:
            return None, "❌ **Mapping Failed:** None of the locations could be found on the map."
        
        hover_kwargs = _hover_kwargs(plot_df, loc_col, '__lat__', '__lon__')
        # Hide the secret translation columns from the user's hover tooltip
        hover_kwargs["hover_data"]["__lat__"] = False
        hover_kwargs["hover_data"]["__lon__"] = False

        # Dynamic zoom and center
        center_lat = plot_df['__lat__'].mean()
        center_lon = plot_df['__lon__'].mean()
        lat_range = plot_df['__lat__'].max() - plot_df['__lat__'].min()
        lon_range = plot_df['__lon__'].max() - plot_df['__lon__'].min()
        max_range = max(lat_range, lon_range)
        if max_range == 0: zoom = 6  # Single point (e.g. 1 state/city), show regional context
        elif max_range > 15: zoom = 4
        elif max_range > 8: zoom = 5
        elif max_range > 4: zoom = 6
        elif max_range > 2: zoom = 7
        elif max_range > 1: zoom = 8
        elif max_range > 0.5: zoom = 9
        elif max_range > 0.2: zoom = 10
        elif max_range > 0.1: zoom = 11
        elif max_range > 0.05: zoom = 12
        else: zoom = 13

        geojson_path = "districts_aot.geojson"
        geojson_data = None
        if os.path.exists(geojson_path):
            try:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)
            except Exception:
                pass

        if geojson_data:
            # Capitalize locations to match GeoJSON properties (usually title case)
            plot_df['__match_loc__'] = plot_df[loc_col].astype(str).str.title()
            
            # Verify if ANY locations actually match the GeoJSON features
            valid_features = {f.get('properties', {}).get('NAME_2', '').title() for f in geojson_data.get('features', [])}
            match_count = plot_df['__match_loc__'].isin(valid_features).sum()

            if match_count > 0:
                # Override hover kwargs for choropleth
                hover_kwargs = _hover_kwargs(plot_df, loc_col)
                
                fig = px.choropleth_mapbox(
                    plot_df,
                    geojson=geojson_data,
                    locations='__match_loc__',
                    featureidkey="properties.NAME_2",
                    color=color_col if color_col else loc_col,
                    color_discrete_sequence=PALETTE,
                    mapbox_style="white-bg",
                    zoom=zoom,
                    center={"lat": center_lat, "lon": center_lon},
                    opacity=0.6,
                    **hover_kwargs
                )
            else:
                # Fallback to scatter map if the dataset locations (e.g. States) don't match our District GeoJSON
                geojson_data = None

        if not geojson_data:
            fig = px.scatter_mapbox(
                plot_df, lat='__lat__', lon='__lon__', color=color_col, size=size_col,
                color_discrete_sequence=PALETTE, mapbox_style="white-bg", zoom=zoom,
                center={"lat": center_lat, "lon": center_lon},
                **hover_kwargs
            )
            
        fig.update_layout(
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google Maps Satellite",
                    "source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"]
                }
            ],
            margin={"r":0,"t":0,"l":0,"b":0},
            hovermode="closest"
        )
        return fig, warning_msg
    
    return None, None


def _bar(df, x_col, y_col, color_col, cfg) -> go.Figure:
    hover_cat = color_col if color_col else x_col
    hover_kwargs = _hover_kwargs(df, hover_cat, x_col, y_col)

    fig = px.bar(
        df, x=x_col, y=y_col,
        color=color_col,
        text_auto=".3s",
        color_discrete_sequence=PALETTE,
        labels={
            x_col: cfg.get("x_label", _prettify(x_col)),
            y_col: cfg.get("y_label", _prettify(y_col)) if y_col else "",
        },
        **hover_kwargs,
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=11,
        marker_line_width=0,
        cliponaxis=False,
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        bargap=0.25,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        hovermode="closest",
    )
    return fig


def _pie(df, x_col, y_col) -> go.Figure:
    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        hover_kwargs = _hover_kwargs(df, x_col, y_col)
        fig = px.pie(
            df, names=x_col, values=y_col, hole=0.42,
            color_discrete_sequence=PALETTE,
            **hover_kwargs,
        )
    else:
        # Auto-aggregate if no numeric value column provided
        agg = df[x_col].value_counts().reset_index()
        agg.columns = [x_col, "count"]
        fig = px.pie(
            agg, names=x_col, values="count", hole=0.42,
            color_discrete_sequence=PALETTE,
            hover_name=x_col,
            hover_data={x_col: False},
        )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        insidetextorientation="radial",
        pull=[0.03] * len(df),
    )
    return fig


def _line(df, x_col, y_col, color_col, cfg) -> go.Figure:
    hover_cat = color_col if color_col else x_col
    hover_kwargs = _hover_kwargs(df, hover_cat, x_col, y_col)

    return px.line(
        df, x=x_col, y=y_col,
        color=color_col,
        markers=True,
        color_discrete_sequence=PALETTE,
        labels={
            x_col: cfg.get("x_label", _prettify(x_col)),
            y_col: cfg.get("y_label", _prettify(y_col)) if y_col else "",
        },
        **hover_kwargs,
    )


def _scatter(df, x_col, y_col, color_col, cfg) -> go.Figure:
    hover_cat = color_col if color_col else x_col
    hover_kwargs = _hover_kwargs(df, hover_cat, x_col, y_col)

    return px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        opacity=0.8,
        color_discrete_sequence=PALETTE,
        labels={
            x_col: cfg.get("x_label", _prettify(x_col)),
            y_col: cfg.get("y_label", _prettify(y_col)) if y_col else "",
        },
        **hover_kwargs,
    )


def _histogram(df, x_col) -> go.Figure:
    hover_kwargs = _hover_kwargs(df, x_col)
    return px.histogram(
        df, x=x_col,
        nbins=min(40, df[x_col].nunique()),
        color=x_col if df[x_col].nunique() < 20 else None,
        color_discrete_sequence=PALETTE,
        labels={x_col: _prettify(x_col)},
        **hover_kwargs,
    )


def _treemap(df, x_col, y_col) -> go.Figure:
    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        hover_kwargs = _hover_kwargs(df, x_col, y_col)
        return px.treemap(
            df, path=[x_col], values=y_col,
            color=x_col, 
            color_discrete_sequence=PALETTE,
            **hover_kwargs,
        )
    agg = df[x_col].value_counts().reset_index()
    agg.columns = [x_col, "count"]
    return px.treemap(
        agg, path=[x_col], values="count",
        color=x_col,
        color_discrete_sequence=PALETTE,
        hover_name=x_col,
        hover_data={x_col: False},
    )


def _funnel(df, x_col, y_col) -> go.Figure:
    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        hover_kwargs = _hover_kwargs(df, x_col, y_col)
        return px.funnel(
            df, y=x_col, x=y_col,
            color=x_col,
            color_discrete_sequence=PALETTE,
            **hover_kwargs,
        )
    # Fall back to horizontal bar
    hover_kwargs = _hover_kwargs(df, x_col)
    return px.bar(
        df, y=x_col, orientation="h",
        color=x_col,
        color_discrete_sequence=PALETTE,
        **hover_kwargs,
    )


# ─────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────

def _resolve_col(
    df: pd.DataFrame,
    preferred: Optional[str],
    cols: list,
    fallback_idx: int,
) -> Optional[str]:
    """Returns the preferred column if it exists; otherwise falls back by position."""
    if preferred and preferred in cols:
        return preferred
    if fallback_idx < len(cols):
        return cols[fallback_idx]
    return None


def _auto_correct_chart_type(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str],
    requested: str,
) -> str:
    """
    Overrides a requested chart type if it would produce a broken/ugly result.
    """
    n_cats = df[x_col].nunique() if x_col in df.columns else 0

    if requested == "pie" and n_cats > 10:
        return "bar"
    if requested == "histogram":
        if x_col in df.columns and not pd.api.types.is_numeric_dtype(df[x_col]):
            return "bar"
    return requested


def _prettify(col_name: str) -> str:
    """Converts snake_case column names to Title Case for axis labels."""
    return col_name.replace("_", " ").title() if col_name else ""


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
    """Applies a consistent, clean, transparent layout to every chart."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 15, "color": BRAND_BLUE, "family": FONT_FAMILY},
        },
        font={"family": FONT_FAMILY, "size": 12, "color": "#2F84D8"},
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(200,200,200,0.5)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family=FONT_FAMILY,
            font_color="black",
            bordercolor="#1F618D",
            align="left",
        ),
        hovermode="closest",
    )
    for axis in ("xaxis", "yaxis"):
        if hasattr(fig.layout, axis):
            getattr(fig.layout, axis).update(
                gridcolor="rgba(200,200,200,0.3)",
                zerolinecolor="rgba(200,200,200,0.5)",
            )
    return fig


# ─────────────────────────────────────────────────────────
# 6. SIDEBAR SETTINGS (ACCESSIBILITY FOCUSED)
# ─────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Renders a minimal settings sidebar focused on visibility and accessibility."""
    with st.sidebar:
        st.markdown("### ⚙️ View Settings")

        text_size = st.radio(
            "🔠 Text Size",
            ["Normal", "Large"],
            horizontal=True,
            key="sidebar_text_size_toggle"
        )

        high_contrast = st.toggle(
            "🌗 High Contrast Mode",
            value=False,
            key="sidebar_contrast_toggle"
        )

    custom_css = ""

    if text_size == "Large":
        custom_css += """
        html, body, [class*="css"] { font-size: 1.2rem !important; }
        .stDataFrame { font-size: 1.1rem !important; }
        """

    if high_contrast:
        custom_css += """
        /* 1. Force Pure White Backgrounds */
        .stApp, [data-testid="stHeader"], [data-testid="stSidebar"] { 
            background-color: #FFFFFF !important; 
        }
        
        /* 2. Force Absolute Black Text Everywhere */
        h1, h2, h3, h4, h5, h6, p, label, div, span, li { 
            color: #000000 !important; 
        }
        
        /* 3. 🚨 OVERRIDE THE HTML TABLE DATA 🚨 */
        [data-testid="stTable"] { background-color: #FFFFFF !important; }
        [data-testid="stTable"] table { 
            border: 3px solid #000000 !important; 
        }
        [data-testid="stTable"] th { 
            background-color: #FFFFFF !important; 
            color: #000000 !important; 
            border-bottom: 3px solid #000000 !important; 
            font-weight: 900 !important; 
            font-size: 1.1rem !important;
        }
        [data-testid="stTable"] td { 
            background-color: #FFFFFF !important; 
            color: #000000 !important; 
            border-bottom: 1px solid #000000 !important; 
            font-weight: 700 !important; 
        }
        
        /* 4. Thick, Black Borders for Buttons & Uploaders */
        .stButton > button, .stDownloadButton > button { 
            border: 3px solid #000000 !important; 
            color: #000000 !important; 
            background-color: #FFFFFF !important; 
            font-weight: 900 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { 
            background-color: #F0F0F0 !important; 
            color: #000000 !important; 
            border-color: #000000 !important;
        }
        [data-testid="stFileUploadDropzone"] { 
            background-color: #FFFFFF !important; 
            border: 3px dashed #000000 !important; 
        }
        
        /* 5. High Contrast Metrics & Lines */
        [data-testid="stMetricValue"] { 
            color: #000000 !important; 
            font-weight: 900 !important; 
        }
        hr { 
            border-bottom: 3px solid #000000 !important; 
        }
        """

    if custom_css:
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# 7. CUSTOM ANIMATED LOADER (PURE CONTRAST)
# ─────────────────────────────────────────────────────────

@contextlib.contextmanager
def farm_loader(text=""):
    """
    Phase 1 & 2: Full-screen blocking overlay loader.
    """
    placeholder = st.empty()
    html_code = f"""
    <style>
    .full-screen-loader {{
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 999999;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        gap: 12px;
        
        /* Clean Professional Frosted Glass Effect */
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
    }}
    .farm-emoji-full {{
        font-size: 2.5rem;
        filter: drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.15));
        animation: full-bounce 1.2s infinite ease-in-out both;
    }}
    .farm-emoji-full:nth-child(1) {{ animation-delay: -0.6s; }}
    .farm-emoji-full:nth-child(2) {{ animation-delay: -0.4s; }}
    .farm-emoji-full:nth-child(3) {{ animation-delay: -0.2s; }}
    .farm-emoji-full:nth-child(4) {{ animation-delay: 0s; }}

    @keyframes full-bounce {{
        0%, 80%, 100% {{ transform: translateY(0) scale(0.9); opacity: 0.8; }}
        40% {{ transform: translateY(-12px) scale(1.1); opacity: 1; }}
    }}
    </style>
    <div class="full-screen-loader">
        <div style="display: flex; gap: 12px;">
            <div class="farm-emoji-full">🌾</div>
            <div class="farm-emoji-full">🤖</div>
            <div class="farm-emoji-full">📊</div>
            <div class="farm-emoji-full">🍃</div>
        </div>
        {f'<div style="font-size: 0.95rem; margin-top: 10px; font-weight: 500; color: #4b5563; text-align: center; white-space: pre-wrap; font-family: sans-serif;">{text}</div>' if text else ''}
    </div>
    """
    placeholder.markdown(html_code, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


@contextlib.contextmanager
def inline_farm_loader(text="AI is analysing your data and building the chart...", auto_scroll_down=False):
    """
    Phase 3: Smooth inline non-blocking loader shown while AI generates chart.
    """
    placeholder = st.empty()
    if auto_scroll_down:
        import time
        ts = int(time.time() * 1000)
        st.components.v1.html(
            f"""
            <script>
            setTimeout(function() {{
                const targets = [
                    window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                    window.parent.document.querySelector('.main'),
                    window.parent.document.documentElement,
                    window.parent.document.body
                ];
                targets.forEach(t => {{ if (t) {{ t.scrollBy({{top: 500, behavior: 'smooth'}}); }} }});
            }}, 50);
            </script>
            """, height=0
        )
        
    html_code = f"""
    <style>
    .inline-loader {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        gap: 8px;
        padding: 15px 0px;
    }}
    .inline-loader-row {{
        display: flex;
        gap: 10px;
        font-size: 1.5rem;
    }}
    .farm-emoji-inline {{
        filter: drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.2));
        animation: inline-bounce 1.2s infinite ease-in-out both;
    }}
    .farm-emoji-inline:nth-child(1) {{ animation-delay: -0.6s; }}
    .farm-emoji-inline:nth-child(2) {{ animation-delay: -0.4s; }}
    .farm-emoji-inline:nth-child(3) {{ animation-delay: -0.2s; }}
    .farm-emoji-inline:nth-child(4) {{ animation-delay: 0s; }}

    @keyframes inline-bounce {{
        0%, 80%, 100% {{ transform: translateY(0) scale(0.9); opacity: 0.7; }}
        40% {{ transform: translateY(-8px) scale(1.15); opacity: 1; }}
    }}
    </style>
    <div class="inline-loader" id="active_farm_loader">
        <div class="inline-loader-row">
            <div class="farm-emoji-inline">&#127806;</div>
            <div class="farm-emoji-inline">&#129302;</div>
            <div class="farm-emoji-inline">&#128202;</div>
            <div class="farm-emoji-inline">&#127811;</div>
        </div>
        <div style="font-size: 0.95rem; font-weight: 500; color: #6b7280; font-family: sans-serif; margin-top: 5px;">
            {text}
        </div>
    </div>
    """
    placeholder.markdown(html_code, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()