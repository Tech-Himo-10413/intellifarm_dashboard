"""
main.py
=======
Data Roots: Insight Studio
3-Phase Architecture: Upload → Clean → Dashboard Studio

Security hardened, session-state safe, and focused on
AI-driven dashboard generation from natural language queries.
"""

import io
import os
import time
import subprocess

import requests
import streamlit as st
import pandas as pd
import polars as pl

import data_manager as dm
import llm_helper as llm_mod
import ui_components as ui
from ui_components import apply_custom_css, render_sidebar, farm_loader, inline_farm_loader
import sys
import asyncio

# Fix for [WinError 10054] on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the VERY FIRST Streamlit command
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliFarm Dashboard",
    page_icon="🌾📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MAX_FILE_MB = 50
ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls"]
DEFAULT_MODEL = "qwen2.5-coder:7b"

# ─────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────
_DEFAULTS = {
    "raw_df": None,           # Polars DataFrame from upload
    "raw_bytes": None,        # Raw bytes for change detection
    "raw_filename": None,
    "health_report": None,    # Dict from generate_health_report()
    "cleaned_df": None,       # Polars DataFrame after Phase 2
    "corrections_done": False,
    "llm": None,              # LLM config dict {model, url}
    "dashboard_results": None,# Last AI dashboard output
    "duckdb_conn": None,      # DuckDB in-memory connection
    "switch_to_tab": None,    # Tab auto-switch index
    "pending_file_bytes": None,          # Held so a missing-dependency fix can retry without re-upload
    "pending_filename": None,
    "pending_missing_dependency": None,  # e.g. "openpyxl" while waiting on the one-click install
    "pending_skip_rows": 0,              # skip_n at the time the pending file was set aside, so the
                                          # dependency-install retry doesn't silently reset it to 0
    "upload_error": None,                # Persisted (non-ephemeral) upload/read error message, so it
                                          # survives reruns instead of flashing once via st.error + st.stop()
    "_last_upload_identity": None,       # (filename, size) of the last file seen, used to detect a
                                          # genuinely NEW upload so skip_n can be reset for it
}


def _init_state() -> None:
    for key, val in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_workflow() -> None:
    """Clears all derived data when a new file is uploaded."""
    for key in ("raw_df", "health_report", "cleaned_df", "corrections_done",
                "dashboard_results", "duckdb_conn"):
        st.session_state[key] = _DEFAULTS[key]


# ─────────────────────────────────────────────────────────
# CELL-LEVEL TYPE-MISMATCH DETECTION (Phase 2 helper)
# ─────────────────────────────────────────────────────────

def _find_bad_cells(pdf: pd.DataFrame, mismatch_cols: list) -> pd.DataFrame:
    """
    Scans the flagged 'looks numeric but is text' columns and returns the
    EXACT cells (row index, column, value) that fail numeric conversion,
    so the user knows precisely what to fix instead of just which column.
    """
    rows = []
    for col in mismatch_cols:
        if col not in pdf.columns:
            continue
        for idx, val in pdf[col].items():
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s == "":
                continue
            try:
                float(s)
            except (TypeError, ValueError):
                rows.append({"Row #": idx, "Column": col, "Value": val})
    return pd.DataFrame(rows)


FLAG_PREFIX = "🚩 "


def _mark_bad_cells(pdf: pd.DataFrame, bad_cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of pdf with flagged bad cells prefixed by a red-flag marker
    directly in the cell value, so they're visibly obvious inside the single
    editable data_editor grid (data_editor has no cell-background styling support).
    """
    marked = pdf.copy()
    if bad_cells_df.empty:
        return marked
    for _, row in bad_cells_df.iterrows():
        r, c = row["Row #"], row["Column"]
        if r in marked.index and c in marked.columns:
            marked.at[r, c] = f"{FLAG_PREFIX}{marked.at[r, c]}"
    return marked


def _strip_flag_prefix(pdf: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Removes the 🚩 marker prefix before casting/locking data, in case the user left it in place."""
    cleaned = pdf.copy()
    for c in cols:
        if c in cleaned.columns:
            cleaned[c] = cleaned[c].apply(
                lambda v: v[len(FLAG_PREFIX):] if isinstance(v, str) and v.startswith(FLAG_PREFIX) else v
            )
    return cleaned


def _safe_for_display(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Makes any DataFrame safe to hand to st.table()/st.dataframe(), for ANY
    dataset — not just this one.

    Root cause this guards against: pandas' .describe(include="all") mixes
    genuinely different Python types within a single column when a
    non-numeric column is involved — e.g. count/unique/freq come back as
    int, while 'top' comes back as the original string, all in the same
    "object"/"str"-dtype column. When Streamlit serializes that to Arrow for
    rendering, PyArrow samples the early (numeric) values, infers int64,
    then crashes the moment it hits the actual string value (e.g.
    "Could not convert 'palm tree' ... tried to convert to int64").

    This isn't specific to any particular column name or dataset — it can
    happen with ANY .describe(), any mixed-type aggregation, on any project.
    Casting every cell to a plain string first guarantees a single,
    consistent type per column, so Arrow never has to guess.

    Note: on pandas' newer string dtype, real missing values remain
    genuinely NaN even after .astype(str) (only non-missing cells get
    stringified) — so .fillna() still needs to run AFTER astype(str) to
    catch them; running it before does nothing.
    """
    return pdf.astype(str).fillna("—")


def _format_kpi_value(val) -> str:
    """
    Formats a single scalar (KPI card) value for display: thousands
    separators for numbers, plain string otherwise.

    IMPORTANT: this logic must live in a real Python function, NOT inline
    inside an f-string format spec. `{val:, if isinstance(val, (int, float))
    else val}` is NOT valid Python — everything after the `:` in an f-string
    is parsed as a literal format spec, and a conditional expression is not
    a legal format spec. That raised `ValueError: Invalid format specifier`
    every single time a query returned a single scalar (e.g. any "how many
    X are there?" / COUNT(*) question) — which silently killed the whole
    dashboard render for the single most common query type in this app.
    """
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, float):
        return f"{val:,.2f}" if not val.is_integer() else f"{int(val):,}"
    return str(val)


# ─────────────────────────────────────────────────────────
# OLLAMA STARTUP
# ─────────────────────────────────────────────────────────

def _ensure_ollama_running() -> bool:
    """
    Checks if Ollama is reachable. If not, tries to start it in the background.
    Returns True if Ollama is available.
    """
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except requests.ConnectionError:
        try:
            kwargs: dict = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["ollama", "serve"], **kwargs)
            time.sleep(3)
            requests.get("http://localhost:11434", timeout=3)
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────
def scroll_to_top():
    """Injects JS to smoothly scroll the page to the top."""
    js = """
    <script>
    var parent = window.parent.document;
    var container = parent.querySelector('.main .block-container');
    if (container) {
        container.scrollTop = 0;
    }
    // Fallback for browsers that scroll the window object
    window.parent.scrollTo({ top: 0, behavior: 'smooth' });
    </script>
    """
    st.iframe(js, height=1)


_init_state()
ui.apply_custom_css()
ui.render_sidebar()
ollama_alive = _ensure_ollama_running()

if st.session_state.llm is None and ollama_alive:
    llm_cfg, llm_err = llm_mod.get_llm(model=DEFAULT_MODEL)
    if llm_cfg:
        st.session_state.llm = llm_cfg
    else:
        st.sidebar.warning("🌾 The AI assistant isn't ready yet — dashboards can still be viewed, but 'Ask a Question' won't work right now.")
        with st.sidebar.expander("Technical details"):
            st.caption(llm_err)
elif not ollama_alive:
    st.sidebar.warning("🌾 The AI assistant isn't running yet — dashboards can still be viewed, but 'Ask a Question' won't work right now.")
    with st.sidebar.expander("Technical details"):
        st.caption("Ollama not found. Install from https://ollama.ai then run: `ollama run qwen2.5-coder:3b`")

# ── AI Model settings — tucked away as "Advanced", collapsed by default ──
# Model names, memory percentages, and download progress bars are
# meaningless (and confusing) to a farmer just trying to see their crop
# data. This still exists for whoever manages/maintains the app, but it no
# longer sits in plain view — it's one click away, not the first thing seen.
if ollama_alive and st.session_state.llm is not None:
    with st.sidebar:
        with st.expander("⚙️ Advanced (Technical) Settings", expanded=False):
            st.markdown("### 🤖 AI Model")
            available_models = llm_mod.list_available_models()
            current_model = st.session_state.llm["model"]
            options = available_models if current_model in available_models else [current_model] + available_models
            selected_model = st.selectbox(
                "Model powering dashboard generation",
                options,
                index=options.index(current_model),
                help="Getting out-of-memory errors? Switch to a smaller model here, "
                     "e.g. a '1.5b' or '3b' tag uses far less RAM than '7b'. "
                     "Pull one first with: ollama pull qwen2.5-coder:3b",
                key="model_picker",
            )
            if selected_model != current_model:
                st.session_state.llm["model"] = selected_model
                st.rerun()

            # ── Proactive low-memory warning ──
            # Catches the "won't fit" case at page-load time instead of only
            # after the user waits through a doomed generation attempt.
            available_gb = llm_mod.get_available_memory_gb()
            needed_gb = llm_mod.estimate_model_memory_gb(selected_model)
            if available_gb is not None and needed_gb is not None:
                if needed_gb > available_gb * 0.85:
                    st.warning(
                        f"⚠️ '{selected_model}' needs ~{needed_gb} GB to load, but only "
                        f"~{available_gb} GB RAM is free. Generation will likely fail — "
                        f"pick a smaller model above, or close other applications first."
                    )
                else:
                    st.caption(f"✅ ~{available_gb} GB RAM free — should comfortably fit '{selected_model}' (~{needed_gb} GB needed).")

            # ── One-click model download — no terminal required ──
            st.markdown("---")
            st.caption("Need a smaller model? Download one directly — no terminal needed:")
            LIGHT_MODELS = ["qwen2.5-coder:1.5b", "qwen2.5-coder:0.5b", "qwen2.5-coder:3b"]
            pull_choice = st.selectbox(
                "Model to download",
                [m for m in LIGHT_MODELS if m not in available_models] or ["(all lightweight options already installed)"],
                key="pull_choice",
                label_visibility="collapsed",
            )
            if pull_choice in LIGHT_MODELS and st.button(f"⬇️ Download {pull_choice}", width="stretch"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                try:
                    for update in llm_mod.pull_model_stream(pull_choice):
                        status = update.get("status", "")
                        total = update.get("total", 0)
                        completed = update.get("completed", 0)
                        if total:
                            progress_bar.progress(min(completed / total, 1.0))
                            status_text.caption(f"{status} — {completed / (1024**2):.0f} MB / {total / (1024**2):.0f} MB")
                        else:
                            status_text.caption(status)
                    status_text.success(f"✅ {pull_choice} downloaded! Select it above to use it.")
                    st.rerun()
                except Exception as exc:
                    status_text.error(f"❌ Download failed: {exc}")

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#1F618D; margin-bottom:0;'>"
    "🌾📊 IntelliFarm Dashboard</h1>"
    "<p style='text-align:center; color:#7F8C8D; margin-top:4px; font-size:0.95rem;'>"
    "Upload &nbsp;→&nbsp; Clean &nbsp;→&nbsp; Generate Dashboards with AI</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────
# INITIAL APP BOOT SEQUENCE
# ─────────────────────────────────────────────────────────
if "app_has_booted" not in st.session_state:
    # Show the loader for 2 seconds on the very first page load
    with farm_loader():
        time.sleep(2.0)
    # Save the state so it doesn't run again unless they refresh the browser
    st.session_state.app_has_booted = True
    st.rerun()

# ─────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📂  1. Upload Data",
    "🧹  2. Clean & Validate",
    "📊  3. Dashboard Studio",
])

# ══════════════════════════════════════════════════════════
# PHASE 1 — UPLOAD & HEALTH CHECK
# ══════════════════════════════════════════════════════════
with tab1:
    uploaded_file = st.file_uploader(
        f"Upload your dataset (CSV or Excel, max {MAX_FILE_MB} MB)",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=False,
    )

    if uploaded_file:
        # 1. Detect if this is a GENUINELY new file (to reset skip_n to 0)
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        is_new_file = st.session_state.get("_last_file_id") != file_id
        
        if is_new_file:
            st.session_state["skip_n_input"] = 0  # Force widget reset to 0
            st.session_state["_last_file_id"] = file_id
            st.session_state["_last_skip_n"] = None # Force a data reload below
            _reset_workflow()
            st.session_state.raw_filename = uploaded_file.name
            st.session_state.upload_error = None

        # ── Raw Preview & Header Row Setup ──
        with st.expander("👀 Raw Preview & Header Row Setup", expanded=True):
            st.caption(
                "This is your file exactly as-is, with no processing — use it to see which "
                "row your real column headers are on, then set the number below."
            )
            preview_bytes = uploaded_file.getvalue()
            prev_df, prev_err = dm.preview_raw_rows(preview_bytes, uploaded_file.name, n_rows=8)

            if prev_err and prev_err.startswith(dm.MISSING_DEPENDENCY_PREFIX):
                st.info(
                    "🌾 The raw preview needs the same one-click Excel setup step as the main "
                    "loader below — the file will still load normally once that's done."
                )
            elif prev_err:
                st.warning(f"Couldn't build a raw preview: {prev_err}")
            elif prev_df is not None:
                st.dataframe(prev_df, width="stretch")

            skip_n = st.number_input(
                "Skip top N rows (rows above your real column headers)",
                min_value=0, step=1,
                help="Count using the 'Row N' labels in the preview above.",
                key="skip_n_input",
            )

        # 2. Detect if skip_n changed, OR if it's a new file.
        if st.session_state.get("_last_skip_n") != skip_n:
            st.session_state["_last_skip_n"] = skip_n
            
            with farm_loader():
                file_bytes = uploaded_file.getvalue()
                if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
                    st.session_state.upload_error = f"❌ File '{uploaded_file.name}' is too large."
                else:
                    raw_df, err = dm.load_file(file_bytes, uploaded_file.name, skip_rows=skip_n)

                    if err and err.startswith(dm.MISSING_DEPENDENCY_PREFIX):
                        st.session_state.pending_file_bytes = file_bytes
                        st.session_state.pending_filename = uploaded_file.name
                        st.session_state.pending_missing_dependency = err.split(":", 1)[1]
                        st.session_state.pending_skip_rows = skip_n
                    elif err or raw_df is None:
                        st.session_state.upload_error = f"🌾 Could not read with **{skip_n}** row(s) skipped: {err}"
                    else:
                        raw_df = dm.sanitize_columns(raw_df)
                        for col in raw_df.columns:
                            if raw_df[col].dtype == 'object':
                                raw_df[col] = raw_df[col].astype(str)

                        st.session_state.raw_df = raw_df
                        st.session_state.health_report = dm.generate_health_report(raw_df)
                        st.session_state.upload_error = None # Clear errors on success

                time.sleep(0.5)
            
            # SLIDE DOWN FIX: Set a flag to scroll down smoothly after the app reloads
            if not is_new_file:
                st.session_state.scroll_after_skip = True
            st.rerun()

    # ── Persisted upload/read error ──
    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)

    # ── One-time setup for Excel support, if this file format needs it ──
    if st.session_state.get("pending_missing_dependency"):
        pkg = st.session_state.pending_missing_dependency
        st.warning(
            "🌾 This app needs one small extra piece to read this kind of Excel file — "
            "a one-time setup step that takes just a few seconds. It isn't a problem "
            "with your file."
        )
        if st.button("🔧 Set Up Excel Support", type="primary"):
            with st.spinner("Setting up Excel support…"):
                ok, install_err = dm.ensure_package(pkg)
            if ok:
                raw_df, err = dm.load_file(
                    st.session_state.pending_file_bytes,
                    st.session_state.pending_filename,
                    skip_rows=st.session_state.get("pending_skip_rows", 0),
                )
                if err or raw_df is None:
                    st.session_state.upload_error = f"🌾 Still couldn't read the file after setup: {err}"
                else:
                    raw_df = dm.sanitize_columns(raw_df)
                    st.session_state.raw_df = raw_df
                    st.session_state.health_report = dm.generate_health_report(raw_df)
                    st.session_state.pending_missing_dependency = None
                    st.session_state.pending_file_bytes = None
                    st.session_state.pending_skip_rows = 0
                    st.success("✅ All set! Loading your file now…")
                    st.rerun()
            else:
                st.error("🌾 Automatic setup didn't work this time. You may need to install it manually.")
                with st.expander("Technical details"):
                    st.code(install_err or "No further details available.")

    # ── Display health snapshot ──
    if st.session_state.raw_df is not None and st.session_state.health_report is not None:
        df: pl.DataFrame = st.session_state.raw_df
        report: dict = st.session_state.health_report

        missing_total = sum(v["count"] for v in report["missing"].values())
        num_numeric = sum(
            1 for dt in df.dtypes
            if dt in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
        )
        num_text = df.width - num_numeric

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🗂 Rows", f"{df.height:,}")
        c2.metric("📐 Columns", f"{df.width}")
        c3.metric("🔢 Numeric", str(num_numeric))
        c4.metric("🏷 Text", str(num_text))
        c5.metric("❓ Missing", f"{missing_total:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        score: int = report.get("health_score", 100)
        score_color = "#27AE60" if score >= 75 else "#F39C12" if score >= 50 else "#E74C3C"
        st.markdown(
            f"<div style='text-align:center; font-size:1rem;'>Data Health Score: "
            f"<strong style='color:{score_color}; font-size:1.2rem;'>{score}/100</strong></div>",
            unsafe_allow_html=True,
        )
        st.progress(score / 100)

        if report["type_mismatches"]:
            with st.expander(f"⚠️ Type Mismatches ({len(report['type_mismatches'])} columns)"):
                for tm in report["type_mismatches"]:
                    st.markdown(f"- **{tm['column']}** looks numeric but is stored as text.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_prev, col_desc, col_types = st.columns(3)

        with col_prev:
            st.markdown("**Preview (Top 10 rows)**")
            st.table(df.head(10).to_pandas())

        with col_desc:
            st.markdown("**Statistical Summary**")
            st.table(_safe_for_display(df.to_pandas().describe(include="all")))

        with col_types:
            st.markdown("**Column Schema**")
            schema_df = pd.DataFrame([
                {"Column": col, "Type": str(dt), "Nulls": df[col].null_count(), "Unique": df[col].n_unique()}
                for col, dt in zip(df.columns, df.dtypes)
            ])
            if not schema_df.empty:
                schema_df = schema_df.set_index("Column")
            st.table(schema_df)

        st.markdown("---")
        st.markdown("### 📈 Quick Number Summary")
        pdf = df.to_pandas()
        numeric_cols = pdf.select_dtypes(include=['number'])

        if not numeric_cols.empty:
            numeric_cols = numeric_cols.fillna(0)
            summary_stats = numeric_cols.describe().T[['min', 'max', 'mean']].round(2)
            summary_stats.columns = ['Lowest (Min)', 'Highest (Max)', 'Average']
            summary_stats = summary_stats.reset_index().rename(columns={'index': 'Data Column'})
            st.dataframe(summary_stats, width="stretch", hide_index=True)
        else:
            st.info("No numeric data found to summarize.")

        st.markdown("---")
        if st.button("Proceed to Data Cleaning  ➡️", type="primary"):
            with farm_loader():
                time.sleep(1.0)
            st.session_state.switch_to_tab = 1
            st.rerun()

    # ── INJECT SLIDE DOWN JAVASCRIPT ──
    if st.session_state.pop("scroll_after_skip", False):
        st.iframe(
            '''<script>
            setTimeout(function() {
                window.parent.scrollBy({ top: 400, left: 0, behavior: 'smooth' });
            }, 300);
            </script>''',
            height=1,
        )


# ══════════════════════════════════════════════════════════
# PHASE 2 — CLEAN & VALIDATE
# ══════════════════════════════════════════════════════════
with tab2:
    if st.session_state.raw_df is None:
        st.info("👈 Upload a file in Step 1 first.")
    else:
        df: pl.DataFrame = st.session_state.raw_df
        report: dict = st.session_state.health_report

        st.markdown("### 🧹 Review & Clean Data")
        issues_found = False
        missing_cols = {c: v for c, v in report["missing"].items() if v["count"] > 0}
        col_issues, col_options = st.columns([1.5, 1])

        with col_issues:
            if missing_cols:
                issues_found = True
                with st.expander(f"⚠️ Missing Values — {len(missing_cols)} column(s) affected", expanded=True):
                    rows = [{"Column": c, "Missing": v["count"], "Pct": f"{v['pct']:.1f}%"} for c, v in missing_cols.items()]
                    missing_df = pd.DataFrame(rows)
                    if not missing_df.empty:
                        missing_df = missing_df.set_index("Column")
                    st.table(missing_df)

            if report.get("duplicate_rows", 0) > 0:
                issues_found = True
                st.warning(f"⚠️ **{report['duplicate_rows']}** duplicate rows detected.")

            mismatch_cols = [tm["column"] for tm in report.get("type_mismatches", [])]
            pandas_df = df.to_pandas()
            bad_cells_df = _find_bad_cells(pandas_df, mismatch_cols) if mismatch_cols else pd.DataFrame()

            if report.get("type_mismatches"):
                issues_found = True
                with st.expander(
                    f"⚠️ Type Mismatches ({len(report['type_mismatches'])} columns, "
                    f"{len(bad_cells_df)} bad cells)",
                    expanded=True,
                ):
                    for tm in report["type_mismatches"]:
                        st.markdown(f"- **`{tm['column']}`**: Looks numeric but is stored as text.")
                    st.caption(
                        "🚩 The exact bad cells are flagged with this marker directly in the "
                        "editable table below — fix them there, or just enable "
                        "'Auto-convert text to numbers' on the right."
                    )

            if not issues_found:
                st.success("✅ No major data quality issues detected.")

        with col_options:
            st.markdown("**Auto-Fix Options**")
            auto_fill = st.checkbox("Auto-fill any remaining empty cells with 0", True)
            drop_dups = st.checkbox("Remove duplicate rows", value=True)
            auto_cast = st.checkbox("Auto-convert text to numbers (Fixes Mismatches)", value=True)

        display_df = _mark_bad_cells(pandas_df, bad_cells_df)

        st.markdown("**Live Data Editor:**")
        if not bad_cells_df.empty:
            st.caption(f"🚩 {len(bad_cells_df)} cell(s) flagged below — edit them directly in the grid.")
        st.caption("💡 Tip: press Enter or click outside a cell before scrolling, to avoid the grid getting stuck mid-edit.")
        edited = st.data_editor(
            display_df,
            num_rows="fixed",  # "dynamic" mode's extra add/remove-row machinery is the more common
                                # trigger for the grid freezing mid-scroll while a cell is being edited.
            width="stretch",
            height=380,
            key=f"data_editor_{st.session_state.get('files_state')}",  # stable per-file key so the
                                                                        # widget remounts cleanly instead
                                                                        # of carrying over stale edit state.
        )
        if mismatch_cols:
            edited = _strip_flag_prefix(edited, mismatch_cols)
        st.markdown("---")

        if st.button("🔒 Lock Data & Go to Dashboard Studio", type="primary"):
            with farm_loader():
                final_df = pl.from_pandas(edited)

                # IMPORTANT: cast BEFORE fill. Any cell the user didn't manually
                # fix (e.g. still "kloi0") only becomes null once cast() runs —
                # so fill must happen AFTER cast, or those error cells never get
                # replaced with 0 and stay silently blank.
                if auto_cast and report.get("type_mismatches"):
                    for tm in report["type_mismatches"]:
                        final_df = final_df.with_columns(pl.col(tm["column"]).cast(pl.Float64, strict=False))

                if auto_fill:
                    final_df = dm.auto_fill_missing(final_df)

                if drop_dups:
                    # maintain_order=True is essential: Polars' unique() reorders
                    # rows by internal hash by default, which was silently
                    # shuffling the user's original row order — confusing and
                    # trust-eroding, especially on large datasets.
                    final_df = final_df.unique(maintain_order=True)

                st.session_state.cleaned_df = final_df
                conn, db_err = dm.create_duckdb_connection(final_df)

                time.sleep(1.5)

            if db_err:
                st.error(f"❌ Database error: {db_err}")
            else:
                st.session_state.duckdb_conn = conn
                st.session_state.corrections_done = True
                st.session_state.switch_to_tab = 2
                st.rerun()

# ══════════════════════════════════════════════════════════
# PHASE 3 — DASHBOARD STUDIO
# ══════════════════════════════════════════════════════════
with tab3:
    cleaned_df = st.session_state.get("cleaned_df")
    conn = st.session_state.get("duckdb_conn")

    # If data is missing, fall back to a safe empty table so nothing crashes.
    if cleaned_df is None:
        cleaned_df = pl.DataFrame()

    if not st.session_state.get("corrections_done") or cleaned_df.is_empty():
        st.warning("🔒 Please clean and lock your data in Step 2 first.")
    else:
        col_view, col_download = st.columns([3, 1])
        with col_view:
            with st.expander("📋 Available Columns (click to expand)"):
                col_ref = pd.DataFrame([
                    {"Column": c, "Type": str(dt), "Unique Values": cleaned_df[c].n_unique(), "Nulls": cleaned_df[c].null_count()}
                    for c, dt in zip(cleaned_df.columns, cleaned_df.dtypes)
                ])
                if not col_ref.empty:
                    st.table(col_ref.set_index("Column"))

        with col_download:
            # Dynamically generate the download filename based on the original upload
            orig_name = st.session_state.get("raw_filename", "dataset")
            # Strip the original extension (.csv, .xlsx) and append _cleaned.csv
            base_name = orig_name.rsplit(".", 1)[0]
            dl_filename = f"{base_name}_cleaned.csv"
            
            st.download_button(
                "💾 Download Data", 
                cleaned_df.write_csv(), 
                dl_filename, 
                "text/csv", 
                width="stretch"
            )

        # ── Quick Insights: always-visible baseline charts ──
        # Permanent, regardless of whether the AI has generated a custom
        # dashboard yet — staying visible across repeated AI generations
        # gives a consistent reference view instead of the layout jumping
        # around each time.
        st.markdown("---")
        st.markdown("### 📊 Quick Insights")
        st.caption("An always-on baseline view of your data. Ask a question below to add AI-generated charts alongside it.")

        pdf = cleaned_df.to_pandas()
        cat_cols = pdf.select_dtypes(include=['object', 'string', 'category']).columns
        num_cols = pdf.select_dtypes(include=['number']).columns

        default_cols = st.columns(2)
        with default_cols[0]:
            if len(cat_cols) > 0:
                c_name = cat_cols[0]
                agg = pdf[c_name].value_counts().reset_index().head(10)
                agg.columns = [c_name, 'Count']
                cfg1 = {
                    "title": f"Top 10: {c_name.replace('_', ' ').title()}",
                    "chart_type": "bar",
                    "x_column": c_name,
                    "y_column": "Count"
                }
                fig1 = ui.build_chart(agg, cfg1)
                if fig1:
                    st.plotly_chart(fig1, width="stretch")
                else:
                    st.warning("Could not render the Bar Chart.")
            else:
                st.info("ℹ️ No text/category columns found to build a Bar Chart.")

        with default_cols[1]:
            if len(num_cols) > 0:
                n_name = num_cols[0]
                cfg2 = {
                    "title": f"Distribution of {n_name.replace('_', ' ').title()}",
                    "chart_type": "histogram",
                    "x_column": n_name
                }
                fig2 = ui.build_chart(pdf, cfg2)
                if fig2:
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.warning("Could not render the Histogram.")
            else:
                st.info("ℹ️ No numeric columns found to build a Histogram.")

        # ── Ask a Question → AI Generates Your Dashboard ──
        st.markdown("---")
        st.markdown("### 💬 Ask a Question → AI Generates Your Dashboard")

        # Build a relevant example from the user's OWN data instead of a
        # generic, unrelated placeholder — much easier for a non-technical
        # user to understand and copy the pattern of.
        _example_pdf = cleaned_df.to_pandas()
        _cat_cols = _example_pdf.select_dtypes(include=['object', 'string', 'category']).columns

        _best_col = None

        # 1. Look for a "good" column (between 2 and 50 unique values, and isn't an "unnamed" index)
        for col in _cat_cols:
            if 2 <= _example_pdf[col].nunique() <= 50 and not col.lower().startswith("unnamed"):
                _best_col = col
                break

        # 2. Fallback: if no perfect match, just pick the first one that isn't "unnamed"
        if not _best_col and len(_cat_cols) > 0:
            _best_col = next((c for c in _cat_cols if not c.lower().startswith("unnamed")), _cat_cols[0])

        if _best_col:
            # Grab a sample value
            _example_vals = _example_pdf[_best_col].dropna().unique()
            _example_val = str(_example_vals[0]) if len(_example_vals) > 0 else "a specific category"

            # Clean up the column name so it looks natural (e.g., 'crop_type_col' -> 'Crop Type')
            _clean_name = _best_col.replace("_col", "").replace("_", " ").title()

            example_hint = f"'How many {_example_val} entries are there?' or 'Show me the top {_clean_name} values'"
        else:
            example_hint = "'How many entries are there?' or 'Show me the highest value'"

        st.caption(f"Examples: *{example_hint}*")

        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_query = st.text_area(
                "Query",
                label_visibility="collapsed",
                placeholder="Type your question in plain language, e.g. \"how many bared soil records are there?\"",
                height=80,
            )
        with btn_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            gen_btn = st.button("📊 Generate", type="primary", width="stretch")
            clr_btn = st.button("🗑️ Clear", width="stretch")

        if clr_btn:
            st.session_state.dashboard_results = None
            st.rerun()

        # AI availability check — guard BEFORE the query is ever sent
        if gen_btn and st.session_state.llm is None:
            st.error("🌾 The AI assistant isn't available right now, so it can't answer this question yet. Please try again in a few minutes.")

        if gen_btn and user_query.strip() and st.session_state.llm is not None:
            st.iframe(
                """
                <script>
                var checkLoader = setInterval(function() {
                    var loader = window.parent.document.querySelector('.inline-loader');
                    if (loader) {
                        loader.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        clearInterval(checkLoader);
                    }
                }, 100);
                setTimeout(function() { clearInterval(checkLoader); }, 2000);
                </script>
                """,
                height=1,
            )
            with inline_farm_loader():
                pdf = cleaned_df.to_pandas()
                text_cols = pdf.select_dtypes(include=['object', 'string', 'category']).columns
                valid_categories = {}
                for col in text_cols:
                    unique_vals = pdf[col].dropna().unique().tolist()[:15]
                    if unique_vals:
                        valid_categories[col] = unique_vals

                corrected_query = llm_mod.correct_user_query(
                    user_query.strip(),
                    valid_categories,
                )

                schema_desc = dm.get_schema_description(cleaned_df)
                result, err = llm_mod.generate_dashboard_config(
                    st.session_state.llm, corrected_query, schema_desc
                )

                time.sleep(0.8)

                if err:
                    st.error(f"❌ {err}")
                elif result:
                    result["corrected_query"] = corrected_query
                    st.session_state.dashboard_results = result

            st.iframe("""<script>
                setTimeout(function() {
                    var inputBox = window.parent.document.querySelector('[data-testid="stTextArea"]');
                    if (inputBox) inputBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 500);
            </script>""", height=1)

        # ── Render AI dashboard results ──
        if st.session_state.get("dashboard_results"):
            dash = st.session_state.dashboard_results
            summary = dash.get("summary", "")
            if summary:
                st.markdown(
                    f"<div class='insight-box'>📌 <strong>AI Insight:</strong> {summary}</div>",
                    unsafe_allow_html=True,
                )

            charts = dash.get("charts", [])
            if not charts:
                sample_cols = ", ".join(ui._prettify(c) for c in list(cleaned_df.columns)[:4])
                st.info(
                    "🌾 I couldn't build a chart from that question. Try being a bit more specific — "
                    f"for example, mention a column name (like **{sample_cols}**), a number, "
                    "or a category value you saw in your data."
                )
            else:
                st.markdown("---")
                for i in range(0, len(charts), 2):
                    row = charts[i: i + 2]
                    grid = st.columns(len(row))

                    for chart_cfg, widget_col in zip(row, grid):
                        with widget_col:
                            sql = chart_cfg.get("sql", "").strip()
                            if not sql:
                                continue

                            result_df, sql_err = dm.execute_sql_query(conn, sql)

                            if sql_err:
                                st.error("🌾 **Chart Generation Failed**", icon="⚠️")
                                st.caption("The local AI model generated a query that syntax errors on this dataset.")
                                with st.expander("Technical details"):
                                    st.code(sql_err, language=None)
                                    st.code(sql, language="sql")

                            elif result_df is not None:
                                pdf_data = result_df.to_pandas()

                                # ─────────────────────────────────────────────────────────
                                # ROBUST RULE: Handle Single-Value / KPI Metric Results
                                # ─────────────────────────────────────────────────────────
                                if pdf_data.shape == (1, 1) or (pdf_data.shape[0] == 1 and pdf_data.shape[1] == 1):
                                    val = pdf_data.iloc[0, 0]
                                    label = chart_cfg.get("title", "Total Value")
                                    display_val = _format_kpi_value(val)

                                    # Render a robust, clean KPI Card instead of failing.
                                    # NOTE: value formatting is done OUTSIDE the f-string via
                                    # _format_kpi_value() — an f-string format spec cannot
                                    # contain a conditional expression, which previously
                                    # crashed this exact block on every COUNT(*)-style query.
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: #F8F9F9; 
                                            padding: 20px; 
                                            border-radius: 10px; 
                                            border-left: 5px solid #27AE60;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                            margin: 10px 0;
                                        ">
                                            <p style="color: #7F8C8D; margin: 0; font-size: 0.9rem; font-weight: bold; text-transform: uppercase;">{label}</p>
                                            <h2 style="color: #2C3E50; margin: 5px 0 0 0; font-size: 2.2rem;">{display_val}</h2>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                # ─────────────────────────────────────────────────────────
                                # Normal Flow: Render plots for valid multi-row/col datasets
                                # ─────────────────────────────────────────────────────────
                                elif not pdf_data.empty and result_df.height > 0:
                                    fig = ui.build_chart(pdf_data, chart_cfg)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Could not render chart layout.")
                                        st.table(pdf_data.head(15))
                                else:
                                    st.info("🌾 No matching data records returned for this metric subset.")
                            else:
                                # Cleaned up dynamic sample text to strictly filter out dirty 'unnamed' names
                                clean_samples = [c for c in list(cleaned_df.columns) if not c.lower().startswith("unnamed")][:3]
                                hint_cols = ", ".join(ui._prettify(c) for c in clean_samples) or "your numeric parameters"
                                st.info(
                                    "🌾 The inquiry returned an un-plottable format. "
                                    f"Try asking for a breakdown over specific factors like **{hint_cols}**."
                                )

# ─────────────────────────────────────────────────────────
# TAB AUTO-SWITCHER & SCROLL-TO-TOP (single, unified — was duplicated before)
# ─────────────────────────────────────────────────────────
if st.session_state.get("switch_to_tab") is not None:
    idx = st.session_state.switch_to_tab

    st.iframe(
        f"""<script>
        (function() {{
            var targetIndex = {idx};
            var attempts = 0;
            var maxAttempts = 40; // retries for ~4 seconds total, in case DOM hasn't mounted yet

            function getDoc() {{
                try {{
                    if (window.parent && window.parent.document && window.parent !== window) {{
                        return window.parent.document;
                    }}
                }} catch (e) {{}}
                return document;
            }}

            function findTabButtons(doc) {{
                var selectors = [
                    'button[data-baseweb="tab"]',
                    '[data-testid="stTabs"] button[role="tab"]',
                    '[role="tablist"] button',
                ];
                for (var i = 0; i < selectors.length; i++) {{
                    var found = doc.querySelectorAll(selectors[i]);
                    if (found && found.length > 0) return found;
                }}
                return null;
            }}

            function fireClick(el) {{
                ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'].forEach(function(type) {{
                    try {{
                        el.dispatchEvent(new MouseEvent(type, {{ bubbles: true, cancelable: true, view: window }}));
                    }} catch (e) {{}}
                }});
                try {{ el.click(); }} catch (e) {{}}
            }}

            function scrollToTop(doc) {{
                var containers = doc.querySelectorAll('.main, [data-testid="stAppViewContainer"], [data-testid="stMain"]');
                containers.forEach(function(c) {{ c.scrollTo({{ top: 0, behavior: 'smooth' }}); }});
                try {{ window.parent.scrollTo({{ top: 0, behavior: 'smooth' }}); }} catch (e) {{}}
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}

            function tryClick() {{
                attempts++;
                var doc = getDoc();
                var tabs = findTabButtons(doc);
                if (tabs && tabs[targetIndex]) {{
                    fireClick(tabs[targetIndex]);
                    setTimeout(function() {{ scrollToTop(doc); }}, 250);
                    return;
                }}
                if (attempts < maxAttempts) {{
                    setTimeout(tryClick, 100);
                }}
            }}

            setTimeout(tryClick, 80);
        }})();
        </script>""",
        height=1,
    )
    st.session_state.switch_to_tab = None
