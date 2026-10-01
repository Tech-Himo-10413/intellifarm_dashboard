import streamlit as st
import pandas as pd
import polars as pl
import data_manager as dm
import ui_components as ui
import llm_helper as llm_mod

def render_phase2():
    if st.session_state.raw_df is None:
        st.info("👈 Upload a file in Step 1 first.")
    else:
        df: pl.DataFrame = st.session_state.raw_df
        report: dict = st.session_state.health_report

        st.markdown("### 🛠️ Review & Clean Data")
        
        # UX Fix: Explicit Narrator Box for Farmers
        st.info(
            "👨‍🌾 **IntelliFarm Assistant:** Welcome to the Data Cleaning step!\n\n"
            "If you see any cells below marked with a red flag (**🚩**), it means the data looks like a number "
            "but is accidentally saved as text (which breaks charts). You can fix them manually in the grid, or simply leave "
            "the **'Auto-convert'** box checked on the right and I'll fix them all for you when you click Lock Data!"
        )

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
            bad_cells_df = dm.find_bad_cells(pandas_df, mismatch_cols) if mismatch_cols else pd.DataFrame()

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

        display_df = dm.mark_bad_cells(pandas_df, bad_cells_df)

        st.markdown("**Live Data Editor:**")
        if not bad_cells_df.empty:
            st.caption(f"🚩 {len(bad_cells_df)} cell(s) flagged below — edit them directly in the grid.")
        st.caption("💡 Tip: press Enter or click outside a cell before scrolling, to avoid the grid getting stuck mid-edit.")
        edited = st.data_editor(
            display_df,
            num_rows="fixed",  
            width="stretch",
            height=380,
            key=f"data_editor_{st.session_state.get('_last_file_id')}",
        )
        if mismatch_cols:
            edited = dm.strip_flag_prefix(edited, mismatch_cols)
        st.markdown("---")

        if st.button("🔒 Lock Data & Go to Dashboard Studio", type="primary"):
            with ui.farm_loader("IntelliFarm AI is locking your data & analysing the dataset to suggest intelligent queries. This might take 10-15 seconds..."):
                final_df = dm.clean_dataset(edited, auto_cast, auto_fill, drop_dups, report)
                st.session_state.cleaned_df = final_df
                conn, db_err = dm.create_duckdb_connection(final_df)
                
                # Pre-generate AI suggestions so Phase 3 doesn't need a second loader
                if "ai_suggestions" not in st.session_state or st.session_state.get("_last_file_id") != st.session_state.get("_sug_file_id"):
                    schema_desc = dm.get_schema_description(final_df)
                    llm_cfg = st.session_state.get("llm")
                    model_name = llm_cfg["model"] if llm_cfg else "qwen2.5-coder:7b"
                    llm, err = llm_mod.get_llm(model_name)
                    if not err and llm:
                        sugs = llm_mod.generate_ai_suggestions(llm, schema_desc)
                        if len(sugs) > 0 and sugs[0] == "Show a summary of the data":
                            st.session_state.ai_toast_warning = "⚠️ Could not generate dynamic AI suggestions. Using defaults."
                        st.session_state.ai_suggestions = sugs
                    else:
                        st.session_state.ai_suggestions = [
                            "Show a summary of the data", 
                            "What are the top categories?", 
                            "Show a pie chart of the distribution",
                            "Show a bar chart of the highest values"
                        ]
                        st.session_state.ai_toast_warning = "⚠️ AI engine offline. Using default suggestions."
                    st.session_state["_sug_file_id"] = st.session_state.get("_last_file_id")

            if db_err:
                st.error(f"❌ Database error: {db_err}")
            else:
                st.session_state.duckdb_conn = conn
                st.session_state.corrections_done = True
                st.session_state.switch_to_tab = 2
                st.session_state.needs_scroll_top = True
                st.rerun()
