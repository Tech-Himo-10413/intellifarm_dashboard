"""
main.py
=======
Data Roots: Insight Studio
3-Phase Architecture: Upload → Clean → Dashboard Studio

Security hardened, session-state safe, and focused on
AI-driven dashboard generation from natural language queries.
"""

import os
import time
import subprocess

import requests
import streamlit as st

import llm_helper as llm_mod
import ui_components as ui
import sys
import asyncio

from phase1_upload import render_phase1
from phase2_clean import render_phase2
from phase3_dashboard import render_phase3

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

# Anchor for scrolling to top
st.markdown("<div id='top_of_page'></div>", unsafe_allow_html=True)

import time
import ui_components as ui
if "app_loaded" not in st.session_state:
    with ui.farm_loader():
        time.sleep(1.5)
    st.session_state.app_loaded = True
    st.rerun()

# ─────────────────────────────────────────────────────────
# DEPLOYMENT CONFIG (env-overridable)
# ─────────────────────────────────────────────────────────
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "50"))
ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls"]
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:7b")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_IS_LOCAL = "localhost" in _OLLAMA_BASE_URL or "127.0.0.1" in _OLLAMA_BASE_URL
ENABLE_MODEL_MANAGEMENT = os.environ.get("ENABLE_MODEL_MANAGEMENT", "true").lower() == "true"

def inject_mobile_styles():
    css_path = os.path.join(os.path.dirname(__file__), "mobile_styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

class SessionManager:
    @staticmethod
    def init():
        defaults = {
            "raw_df": None,           
            "raw_bytes": None,        
            "raw_filename": None,
            "health_report": None,    
            "cleaned_df": None,       
            "corrections_done": False,
            "llm": None,              
            "dashboard_results": None,
            "duckdb_conn": None,      
            "switch_to_tab": None,    
            "upload_error": None,                
            "_last_upload_identity": None,
            "theme": "System Default",
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

def _init_state() -> None:
    SessionManager.init()
    inject_mobile_styles()

def _ensure_ollama_running() -> bool:
    try:
        requests.get(_OLLAMA_BASE_URL, timeout=2)
        return True
    except (requests.ConnectionError, requests.Timeout):
        if not _OLLAMA_IS_LOCAL:
            return False
        try:
            kwargs: dict = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["ollama", "serve"], **kwargs)
            time.sleep(3)
            requests.get(_OLLAMA_BASE_URL, timeout=3)
            return True
        except Exception:
            return False

_init_state()

ui.apply_custom_css()
ui.render_sidebar()
ollama_alive = _ensure_ollama_running()

if st.session_state.llm is None and ollama_alive:
    llm_cfg, llm_err = llm_mod.get_llm(model=DEFAULT_MODEL)
    if llm_cfg:
        st.session_state.llm = llm_cfg
    else:
        st.sidebar.warning("🌾 The AI assistant isn't ready yet.")
elif not ollama_alive:
    st.sidebar.warning("🌾 The AI assistant isn't running yet.")

if ollama_alive and st.session_state.llm is not None and ENABLE_MODEL_MANAGEMENT:
    with st.sidebar:
        with st.expander("⚙️ Advanced Settings", expanded=False):
            available_models = llm_mod.list_available_models()
            current_model = st.session_state.llm["model"]
            options = available_models if current_model in available_models else [current_model] + available_models
            selected_model = st.selectbox(
                "Model powering dashboard generation",
                options,
                index=options.index(current_model),
                key="model_picker",
            )
            if selected_model != current_model:
                st.session_state.llm["model"] = selected_model
                st.rerun()

st.markdown(
    "<h1 style='text-align:center; color:#1F618D; margin-bottom:0;'>"
    "🌾📊 IntelliFarm Dashboard</h1>"
    "<p style='text-align:center; color:#7F8C8D; margin-top:4px; font-size:0.95rem;'>"
    "Upload &nbsp;→&nbsp; Clean &nbsp;→&nbsp; Generate Dashboards with AI</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📂  1. Upload Data",
    "🧹  2. Clean & Validate",
    "📊  3. Dashboard Studio",
])

with tab1:
    render_phase1()

with tab2:
    render_phase2()

with tab3:
    render_phase3()

if st.session_state.pop("needs_scroll_top", False):
    ui.scroll_to_top()

if st.session_state.get("switch_to_tab") is not None:
    idx = st.session_state.switch_to_tab
    st.components.v1.html(
        f"""<script>
        (function() {{
            var targetIndex = {idx};
            var attempts = 0;
            var maxAttempts = 40; 
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
                    try {{ el.dispatchEvent(new MouseEvent(type, {{ bubbles: true, cancelable: true, view: window }})); }} catch (e) {{}}
                }});
                try {{ el.click(); }} catch (e) {{}}
            }}
            function tryClick() {{
                attempts++;
                var doc = getDoc();
                var tabs = findTabButtons(doc);
                if (tabs && tabs[targetIndex]) {{
                    fireClick(tabs[targetIndex]);
                    return;
                }}
                if (attempts < maxAttempts) {{
                    setTimeout(tryClick, 100);
                }}
            }}
            setTimeout(tryClick, 80);
        }})();
        </script>""",
        height=0,
    )
    st.session_state.switch_to_tab = None
