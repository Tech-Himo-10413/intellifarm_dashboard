# 🌾📊 IntelliFarm Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory_SQL-yellow.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)

IntelliFarm Dashboard is a privacy-first, AI-powered analytics studio designed specifically for agricultural data. It allows users to upload raw, messy datasets (CSV/Excel), automatically cleans and validates them, and uses a local Large Language Model (LLM) to generate interactive SQL-driven dashboards via natural language queries.

By leveraging local LLMs through Ollama, IntelliFarm ensures that sensitive agricultural and financial data never leaves the user's machine.

## ✨ Core Features: 3-Phase Architecture

### 📂 Phase 1: Upload & Health Profiling
* **Robust File Ingestion:** Supports `.csv`, `.xlsx`, and `.xls` with automatic encoding fallbacks and header-row detection (skip-rows).
* **Automated Data Profiling:** Instantly generates a Data Health Score, detecting missing values, duplicate rows, statistical outliers, and type mismatches.

### 🧹 Phase 2: Clean & Validate
* **Interactive Data Grid:** View and edit flagged, erroneous cells directly within the UI.
* **Auto-Healing:** One-click auto-filling of missing data, duplicate removal, and text-to-number casting.
* **In-Memory SQL:** Cleansed data is materialized into a highly optimized DuckDB instance for lightning-fast querying.

### 📊 Phase 3: AI Dashboard Studio
* **Text-to-SQL Generation:** Ask questions in plain English (e.g., *"How many bared soil records are there?"*).
* **Resilient Querying:** The backend utilizes an anti-hallucination bridge agent and fuzzy string matching to ensure accurate column mapping before querying DuckDB.
* **Dynamic Plotly Rendering:** Automatically selects the best chart type (Bar, Line, Pie, Treemap, Funnel) and generates visually rich, accessible KPIs and graphs.

## 🛠️ Technology Stack

* **Frontend:** Streamlit, Custom CSS
* **Data Processing:** Polars, Pandas
* **Database:** DuckDB
* **Visualization:** Plotly Express & Graph Objects
* **AI/LLM Integration:** Ollama (Default: `qwen2.5-coder:7b`)

## 🚀 Local Installation

### Prerequisites
1. Ensure Python 3.9+ is installed.
2. Install [Ollama](https://ollama.com/) to run the local AI models.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Tech-Himo-10413/intellifarm_dashboard.git](https://github.com/Tech-Himo-10413/intellifarm_dashboard.git)
   cd intellifarm_dashboard
