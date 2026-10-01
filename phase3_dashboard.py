import streamlit as st
import pandas as pd
import polars as pl
import time
import data_manager as dm
import ui_components as ui
import llm_helper as llm_mod

def render_phase3():
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
            orig_name = st.session_state.get("raw_filename", "dataset")
            base_name = orig_name.rsplit(".", 1)[0] if orig_name else "dataset"
            dl_filename = f"{base_name}_cleaned.csv"
            
            st.download_button(
                "💾 Download Data", 
                cleaned_df.write_csv(), 
                dl_filename, 
                "text/csv", 
                use_container_width=True
            )

        # ── Quick Insights: always-visible baseline charts ──
        st.markdown("---")
        st.markdown("### 📊 Quick Insights")
        st.caption("An always-on baseline view of your data.")

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
                fig1, err1, warn1 = ui.build_chart(agg, cfg1)
                if warn1: st.warning(warn1)
                if err1: st.error(err1)
                elif fig1: st.plotly_chart(fig1, width="stretch", config={'scrollZoom': False})
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
                fig2, err2, warn2 = ui.build_chart(pdf, cfg2)
                if warn2: st.warning(warn2)
                if err2: st.error(err2)
                elif fig2: st.plotly_chart(fig2, width="stretch", config={'scrollZoom': False})
            else:
                st.info("ℹ️ No numeric columns found to build a Histogram.")

        # ── AI Discovery & Data Overview ──
        st.markdown("---")
        st.markdown("### 🌾 AI Data Discovery & Suggestions")
        st.caption("Click any suggestion below to instantly generate a dashboard!")
        
        if "ai_toast_warning" in st.session_state:
            st.toast(st.session_state.pop("ai_toast_warning"), icon="⚠️")
        
        suggestions = st.session_state.get("ai_suggestions", [
            "Show a summary of the data", 
            "What are the top categories?", 
            "Show a pie chart of the distribution",
            "Show a bar chart of the highest values"
        ])

        # Layout chips in a grid (max 3 per row) so they don't get squished
        row_size = 3
        for i in range(0, len(suggestions), row_size):
            cols = st.columns(row_size)
            for j, sug in enumerate(suggestions[i:i+row_size]):
                with cols[j]:
                    if st.button(f"✨ {sug}", use_container_width=True, key=f"sug_{i+j}"):
                        st.session_state.auto_query = sug
                        st.session_state._scroll_to_loader = True
                        st.rerun()

        # ── Ask a Question → AI Generates Your Dashboard ──
        st.markdown("---")
        st.markdown("### 💬 Ask the AI to build your Dashboard")

        q_col, btn_col = st.columns([5, 1])
        with q_col:
            default_q = st.session_state.pop("auto_query", "")
            user_query = st.text_area(
                "Query",
                value=default_q,
                label_visibility="collapsed",
                placeholder="Type your question in plain language, e.g. \"show me a map of the villages\"",
                height=80,
            )
        with btn_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            gen_btn = st.button("📊 Generate", type="primary", use_container_width=True)
            clr_btn = st.button("🗑️ Clear", use_container_width=True)

        if clr_btn:
            st.session_state.dashboard_results = None
            st.rerun()

        # If a chip was clicked, auto-trigger the generate logic
        should_generate = gen_btn or bool(default_q)

        # AI availability check
        if should_generate and st.session_state.llm is None:
            st.warning("🌾 The AI assistant isn't available right now. Please try again in a few minutes.")

        if should_generate and user_query.strip() and st.session_state.llm is not None:
            # Scroll down so user can see the loader — inject JS BEFORE the blocking LLM call
            if st.session_state.pop("_scroll_to_loader", False) or gen_btn:
                import time as _t
                _ts = int(_t.time() * 1000)
                st.components.v1.html(
                    f"""
                    <script>
                    // TS:{_ts}
                    setTimeout(function() {{
                        var main = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                        if (main) {{ main.scrollBy({{top: 500, behavior: 'smooth'}}); }}
                        else {{ window.parent.scrollBy({{top: 500, behavior: 'smooth'}}); }}
                    }}, 100);
                    </script>
                    """, height=0
                )
            with ui.inline_farm_loader():
                time.sleep(0.3)
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

                time.sleep(0.5)

                if err:
                    st.warning(f"{err}")
                elif result:
                    result["corrected_query"] = corrected_query
                    st.session_state.dashboard_results = result
                    if default_q: st.rerun() # Ensure text area updates on auto-submit

        # ── Render AI dashboard results ──
        if st.session_state.get("dashboard_results"):
            dash = st.session_state.dashboard_results
            
            summary = dash.get("summary", "")
            if summary:
                st.markdown(
                    f"<div class='insight-box'>👨‍🌾 <strong>AI Insight:</strong> {summary}</div>",
                    unsafe_allow_html=True,
                )
            
            ai_insight = dash.get("ai_insight", "")
            if ai_insight:
                st.info(f"💡 **Mapping Notice:** {ai_insight}")

            charts = dash.get("charts", [])
            data_sql = dash.get("data_sql", "").strip()

            def _auto_fix_sql(sql: str, error: str, real_cols: list) -> str:
                """
                When DuckDB reports 'column X not found, Candidate bindings: A, B, C',
                fuzzy-match X against the real schema columns and substitute it.
                Handles multiple bad column names in a single pass.
                """
                import re, difflib
                wrong_cols = re.findall(r'Referenced column "([^"]+)" not found', error)
                fixed = sql
                for wrong in wrong_cols:
                    best = difflib.get_close_matches(wrong, real_cols, n=1, cutoff=0.25)
                    if best:
                        fixed = re.sub(rf'\b{re.escape(wrong)}\b', best[0], fixed)
                        fixed = fixed.replace(f'"{wrong}"', f'"{best[0]}"')
                return fixed

            def _render_data_card(result_df, title="Accurate Data Results"):
                """Render query results as a beautiful green-themed data table card."""
                import html
                data_pdf = result_df.to_pandas()
                n_rows = result_df.height
                cols_list = list(data_pdf.columns)
                rows_html = ""
                for idx, row in data_pdf.iterrows():
                    cells = "".join(
                        f'<td style="padding:8px 14px; border-bottom:1px solid #e5e7eb; color:#374151; font-size:0.95rem;">{html.escape(str(v))}</td>'
                        for v in row
                    )
                    bg = "rgba(16,185,129,0.04)" if idx % 2 == 0 else "#ffffff"
                    rows_html += f'<tr style="background:{bg};">{cells}</tr>'
                headers_html = "".join(
                    f'<th style="padding:10px 14px; background:linear-gradient(135deg,#065f46,#047857); color:#ffffff; font-weight:600; font-size:0.9rem; text-align:left; white-space:nowrap;">{html.escape(c.replace("_col","").replace("_"," ").title())}</th>'
                    for c in cols_list
                )
                st.markdown(f"""
<div style="margin-top:16px; border-radius:14px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.10); border:1px solid #d1fae5;">
  <div style="background:linear-gradient(135deg,#065f46,#047857); padding:16px 20px; display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.8rem;">&#128202;</span>
    <div>
      <div style="color:#ffffff; font-weight:700; font-size:1.1rem;">{title}</div>
      <div style="color:#a7f3d0; font-size:0.85rem;">{n_rows} record{'s' if n_rows != 1 else ''} — read directly from your dataset (100% accurate)</div>
    </div>
  </div>
  <div style="overflow-x:auto; background:#ffffff;">
    <table style="width:100%; border-collapse:collapse;">
      <thead><tr>{headers_html}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
""", unsafe_allow_html=True)

            # Get the real column names from the dataset for auto-correction
            real_columns = cleaned_df.columns

            if not charts:
                # No chart generated — try data_sql, auto-fix if it fails, show card
                if data_sql and conn:
                    result, err = dm.execute_sql_query(conn, data_sql)
                    if err:
                        fixed_sql = _auto_fix_sql(data_sql, err, real_columns)
                        if fixed_sql != data_sql:
                            result, err = dm.execute_sql_query(conn, fixed_sql)
                    if not err and result is not None and result.height > 0:
                        _render_data_card(result)
                    else:
                        st.info("🌾 No matching records found in your dataset for that query.")
                else:
                    st.info("🌾 Try rephrasing your question using the exact column names shown in the 'Available Columns' panel above.")
            else:
                st.markdown("---")
                any_chart_rendered = False
                for i in range(0, len(charts), 2):
                    row = charts[i: i + 2]
                    grid = st.columns(len(row))

                    for chart_cfg, widget_col in zip(row, grid):
                        with widget_col:
                            sql = chart_cfg.get("sql", "").strip()
                            if not sql:
                                continue

                            result_df, sql_err = dm.execute_sql_query(conn, sql)

                            # Auto-correct: if column not found, fuzzy-fix and retry
                            if sql_err and "not found" in sql_err:
                                fixed = _auto_fix_sql(sql, sql_err, real_columns)
                                if fixed != sql:
                                    result_df, sql_err = dm.execute_sql_query(conn, fixed)
                                    if not sql_err:
                                        chart_cfg["sql"] = fixed  # use fixed SQL

                            if sql_err:
                                # Chart still failed after auto-fix — try data_sql as fallback display
                                if data_sql and conn:
                                    fb_result, fb_err = dm.execute_sql_query(conn, data_sql)
                                    if fb_err and "not found" in fb_err:
                                        fixed_ds = _auto_fix_sql(data_sql, fb_err, real_columns)
                                        fb_result, fb_err = dm.execute_sql_query(conn, fixed_ds)
                                    if not fb_err and fb_result is not None and fb_result.height > 0:
                                        _render_data_card(fb_result, title=chart_cfg.get("title", "Data Results"))
                                        any_chart_rendered = True
                                        continue
                                # Last resort: show debug
                                with st.expander("&#128269; Chart could not be generated (tap to debug)"):
                                    st.code(sql_err, language=None)
                                    st.code(sql, language="sql")
                            elif result_df is not None:
                                pdf_data = result_df.to_pandas()
                                any_chart_rendered = True

                                if pdf_data.shape == (1, 1) or (pdf_data.shape[0] == 1 and pdf_data.shape[1] == 1):
                                    val = pdf_data.iloc[0, 0]
                                    label = chart_cfg.get("title", "Total Value")
                                    try:
                                        num_val = float(val)
                                        if num_val >= 1_000_000:
                                            display_val = f"{num_val/1_000_000:,.1f}M"
                                        elif num_val >= 1_000:
                                            display_val = f"{num_val:,.0f}"
                                        elif num_val == int(num_val):
                                            display_val = str(int(num_val))
                                        else:
                                            display_val = f"{num_val:,.2f}"
                                    except (TypeError, ValueError):
                                        display_val = str(val)
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
                                elif not pdf_data.empty and result_df.height > 0:
                                    fig, err_msg, warn_msg = ui.build_chart(pdf_data, chart_cfg)
                                    if warn_msg:
                                        st.warning(warn_msg)
                                    if err_msg or fig is None:
                                        # Chart build failed — show as data card instead
                                        _render_data_card(result_df, title=chart_cfg.get("title", "Data Results"))
                                    else:
                                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})
                                else:
                                    st.warning("Could not render chart layout.")
                                    st.table(pdf_data.head(15))
                            else:
                                st.info("&#127806; The inquiry returned an un-plottable format.")


