import streamlit as st
import pandas as pd
import polars as pl
import time
import data_manager as dm
import ui_components as ui

MAX_FILE_MB = 200
ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls"]

def render_phase1():
    st.markdown("### 🌾 Upload Your Farm Data")
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
            
            keys = ["raw_df", "health_report", "cleaned_df", "corrections_done", "dashboard_results", "duckdb_conn"]
            for key in keys:
                st.session_state[key] = None
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
                pkg = prev_err.split(":", 1)[1]
                st.error(f"🌾 Cannot read file. Please install the `{pkg}` package.")
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
            
            with ui.inline_farm_loader(text="Reading your file...", auto_scroll_down=True):
                file_bytes = uploaded_file.getvalue()
                if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
                    st.session_state.upload_error = f"❌ File '{uploaded_file.name}' is too large."
                else:
                    raw_df, err = dm.load_file(file_bytes, uploaded_file.name, skip_rows=skip_n)

                    if err and err.startswith(dm.MISSING_DEPENDENCY_PREFIX):
                        pkg = err.split(":", 1)[1]
                        st.session_state.upload_error = f"❌ Missing dependency. Please run: pip install {pkg}"
                    elif err or raw_df is None:
                        st.session_state.upload_error = f"🌾 Could not read with **{skip_n}** row(s) skipped: {err}"
                    else:
                        raw_df = dm.sanitize_columns(raw_df)

                        st.session_state.raw_df = raw_df
                        st.session_state.health_report = dm.generate_health_report(raw_df)
                        st.session_state.upload_error = None # Clear errors on success

            # SLIDE DOWN FIX: Always scroll down to preview/health report when processing finishes
            st.session_state.scroll_after_skip = True
            st.rerun()

    # ── Persisted upload/read error ──
    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)

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
            st.table(dm.safe_for_display(df.to_pandas().describe(include="all")))

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
            with ui.farm_loader():
                time.sleep(2.5)
            st.session_state.switch_to_tab = 1
            st.session_state.needs_scroll_top = True
            st.rerun()

    # ── INJECT SLIDE DOWN JAVASCRIPT ──
    if st.session_state.pop("scroll_after_skip", False):
        st.components.v1.html(
            """
            <script>
            setTimeout(function() {
                var doc = window.parent.document;
                // Scroll specifically to the data preview tables, or just a solid 600px down
                window.parent.scrollBy({ top: 700, behavior: 'smooth' });
            }, 800);
            </script>
            """,
            height=0
        )
