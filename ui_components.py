"""
ui_components.py
================
Handles visual styling and dynamic Plotly dashboard generation.

Supported chart types: bar, pie, line, scatter, histogram, treemap, funnel.
All builders fall back gracefully if columns are missing or data is sparse.
"""
import contextlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────
BRAND_BLUE = "#1F618D"
PALETTE = [
    "#1F618D", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#3498DB", "#E91E63", "#00BCD4",
]
FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"


# ─────────────────────────────────────────────────────────
# 1. GLOBAL CSS
# ─────────────────────────────────────────────────────────

def apply_custom_css() -> None:
    """Injects dynamic CSS that automatically respects Streamlit's Light/Dark themes."""
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


# ─────────────────────────────────────────────────────────
# 2. MAIN CHART DISPATCHER
# ─────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, cfg: Dict) -> Optional[go.Figure]:
    """
    Builds the best-fitting Plotly chart for the given data and config.
    """
    if df is None or df.empty or df.columns.empty:
        return None

    cols = df.columns.tolist()
    title = cfg.get("title", "Data Insight")
    chart_type = cfg.get("chart_type", "bar").lower()

    # Resolve columns from config, with positional fallbacks
    x_col = _resolve_col(df, cfg.get("x_column"), cols, 0)
    y_col = _resolve_col(df, cfg.get("y_column"), cols, 1)
    color_col = cfg.get("color_column") if cfg.get("color_column") in cols else None

    if x_col is None:
        return None  # Truly cannot determine any column

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
        return _apply_layout(fig, title)

    # ── Auto-correct chart type based on data shape ──
    chart_type = _auto_correct_chart_type(df, x_col, y_col, chart_type)

    # ── Dispatch ──
    try:
        if chart_type == "pie":
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

        return _apply_layout(fig, title)

    except Exception:
        # Last-resort fallback: simple horizontal bar of first two columns
        try:
            fig = px.bar(df, x=y_col, y=x_col, orientation="h",
                         color_discrete_sequence=PALETTE)
            return _apply_layout(fig, title)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────
# 3. HOVER-DETAIL HELPERS
# ─────────────────────────────────────────────────────────
# Every generated chart only reflects the columns the AI's SQL query
# happened to select for x/y/color. Anything else in that result set —
# extra metrics, IDs, secondary breakdowns — was previously invisible on
# hover. These helpers surface the FULL row of underlying data whenever
# the user hovers over a bar, point, or slice, instead of just x & y.

def _dynamic_hover_data(df: pd.DataFrame, shown_cols) -> Dict:
    """
    Builds a plotly-express `hover_data` dict that includes every column
    in the result set NOT already shown elsewhere (axes / hover_name),
    with numeric columns formatted using thousands separators so the
    hover card reads like a clean mini data-card rather than raw numbers.
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
    ready to splice into a plotly.express call so hovering over any chart
    element reveals the full underlying data for that point/bar/slice.
    """
    shown = {primary_col, *other_shown_cols}
    hover_data = _dynamic_hover_data(df, shown)
    hover_data[primary_col] = False  # avoid duplicating the bold hover_name
    return {"hover_name": primary_col, "hover_data": hover_data}


# ─────────────────────────────────────────────────────────
# 4. INDIVIDUAL CHART BUILDERS
# ─────────────────────────────────────────────────────────

def _bar(df, x_col, y_col, color_col, cfg) -> go.Figure:
    # 🟢 Find the best category column to use as the bold hover title
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
        # Native Plotly hover already appends value + percent automatically;
        # combined with hover_data above, the card now shows every extra field too.
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
        color=x_col if df[x_col].nunique() < 20 else None,  # add colors if there aren't too many bars
        color_discrete_sequence=PALETTE,
        labels={x_col: _prettify(x_col)},
        **hover_kwargs,
    )


def _treemap(df, x_col, y_col) -> go.Figure:
    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        hover_kwargs = _hover_kwargs(df, x_col, y_col)
        return px.treemap(
            df, path=[x_col], values=y_col,
            color=x_col,  # forces the treemap to use the custom palette
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
            color=x_col,  # distinct colors per funnel stage
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
    Rules:
      - pie  → switch to bar if more than 10 categories
      - line → switch to bar if X axis is not ordered/sequential
      - histogram → keep only if x_col is numeric
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
        # Styling the Hover Tooltip to look professional and readable
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
    # Subtle grid lines on axes where applicable
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

        # 1. Font Size Toggle
        text_size = st.radio(
            "🔠 Text Size",
            ["Normal", "Large"],
            horizontal=True,
            key="sidebar_text_size_toggle"
        )

        # 2. High Contrast Toggle (Perfect for outdoor visibility)
        high_contrast = st.toggle(
            "🌗 High Contrast Mode",
            value=False,
            key="sidebar_contrast_toggle"
        )

        # The confusing theme message has been removed from here!

    # ── DYNAMIC STYLING LOGIC ──
    custom_css = ""

    # Apply Large Text CSS
    if text_size == "Large":
        custom_css += """
        html, body, [class*="css"] { font-size: 1.2rem !important; }
        .stDataFrame { font-size: 1.1rem !important; }
        """

    # Apply High Contrast CSS
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

    # Inject the CSS into the app
    if custom_css:
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 7. CUSTOM ANIMATED LOADER (PURE CONTRAST)
# ─────────────────────────────────────────────────────────

@contextlib.contextmanager
def farm_loader():
    """
    Phase 1 & 2: Full-screen blocking overlay loader.
    UX UPGRADE: Highly transparent with a soft background blur so the user can see the app working.
    """
    placeholder = st.empty()
    html_code = """
    <style>
    .full-screen-loader {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 999999;
        display: flex; justify-content: center; align-items: center;
        gap: 20px; font-size: 3.5rem;
        
        /* 👇 Transparent Frosted Glass Effect for Full Screen 👇 */
        background-color: rgba(248, 251, 248, 0.35); /* 65% transparent! */
        backdrop-filter: blur(4px); /* Gently blurs the app behind it */
    }
    .farm-emoji-full {
        filter: drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.4));
        animation: full-bounce 1.2s infinite ease-in-out both;
    }
    .farm-emoji-full:nth-child(1) { animation-delay: -0.6s; }
    .farm-emoji-full:nth-child(2) { animation-delay: -0.4s; }
    .farm-emoji-full:nth-child(3) { animation-delay: -0.2s; }
    .farm-emoji-full:nth-child(4) { animation-delay: 0s; }

    @keyframes full-bounce {
        0%, 80%, 100% { transform: translateY(0) scale(0.8); opacity: 0.5; }
        40% { transform: translateY(-25px) scale(1.2); opacity: 1; }
    }
    </style>
    <div class="full-screen-loader">
        <div class="farm-emoji-full">🌾</div>
        <div class="farm-emoji-full">🤖</div>
        <div class="farm-emoji-full">📊</div>
        <div class="farm-emoji-full">🍃</div>
    </div>
    """
    placeholder.markdown(html_code, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


@contextlib.contextmanager
def inline_farm_loader():
    """
    Phase 3: Smooth inline non-blocking loader.
    Reverted to the clean, invisible background style.
    """
    placeholder = st.empty()
    html_code = """
    <style>
    .inline-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        font-size: 2rem;
        padding: 30px 0px;
    }
    .farm-emoji-inline {
        filter: drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.2));
        animation: inline-bounce 1.2s infinite ease-in-out both;
    }
    .farm-emoji-inline:nth-child(1) { animation-delay: -0.6s; }
    .farm-emoji-inline:nth-child(2) { animation-delay: -0.4s; }
    .farm-emoji-inline:nth-child(3) { animation-delay: -0.2s; }
    .farm-emoji-inline:nth-child(4) { animation-delay: 0s; }

    @keyframes inline-bounce {
        0%, 80%, 100% { transform: translateY(0) scale(0.85); opacity: 0.6; }
        40% { transform: translateY(-12px) scale(1.1); opacity: 1; }
    }
    </style>
    <div class="inline-loader">
        <div class="farm-emoji-inline">🌾</div>
        <div class="farm-emoji-inline">🤖</div>
        <div class="farm-emoji-inline">📊</div>
        <div class="farm-emoji-inline">🍃</div>
    </div>
    """
    placeholder.markdown(html_code, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()