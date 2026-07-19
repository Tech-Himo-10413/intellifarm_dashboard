"""
data_manager.py
================
Secure data ingestion, statistical profiling, cleaning, and DuckDB SQL execution.
All public functions return (result, error_string) tuples for clean caller handling.
"""

import io
import re
import sys
import threading
import subprocess
import importlib
import polars as pl
import pandas as pd
import duckdb
from typing import Tuple, Optional, Dict, List, Any

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
STRING_DTYPES = (pl.Utf8, pl.String, pl.Categorical)

NULL_ALIASES = ["", "na", "n/a", "null", "none", "#n/a", "nan", "nil", "-", "--", "?"]

# SQL keywords that should never appear in AI-generated queries
_SQL_BLOCKLIST = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|EXECUTE|TRUNCATE|MERGE|REPLACE)\b",
    re.IGNORECASE,
)

# Prefix used to signal "this file format needs a Python package that isn't
# installed yet" back to the caller, so the UI can offer a one-click fix
# instead of showing a raw ImportError to a non-technical user.
MISSING_DEPENDENCY_PREFIX = "MISSING_DEPENDENCY:"

# Which package each Excel engine needs — used both to detect the failure
# and to know what to install.
_EXCEL_ENGINE_PACKAGES = {
    "xlsx": "openpyxl",
    "xls": "xlrd",
}

# Both openpyxl and xlrd are already pinned in requirements.txt, so in a
# properly built deployment image this fallback should rarely, if ever,
# actually fire — it exists purely as a safety net for slimmed-down or
# hand-rolled environments that skip a full `pip install -r requirements.txt`
# (e.g. a manually assembled container image). Kept for resilience, not
# because it's the expected path.
#
# Multi-user note: ensure_package() shells out to `pip install`, which
# mutates the SHARED Python environment every session in this process runs
# in. Two farmers hitting "Set Up Excel Support" at the same moment would
# otherwise race on the same pip install; this lock serializes that so the
# second caller just waits for the first install to finish instead of both
# corrupting each other's install.
_pip_install_lock = threading.Lock()


# ─────────────────────────────────────────────────────────
# 1. FILE LOADING
# ─────────────────────────────────────────────────────────

def load_file(file_bytes: bytes, filename: str, skip_rows: int = 0) -> Tuple[Optional[pl.DataFrame], str]:
    """
    Loads a CSV or Excel file from raw bytes, skipping garbage rows if requested.
    """
    try:
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            # Pass skip_rows down to the fallback reader
            df = _read_csv_with_encoding_fallback(file_bytes, skip_rows)
        elif ext in ("xlsx", "xls"):
            try:
                # Pandas handles skiprows perfectly
                pandas_df = pd.read_excel(io.BytesIO(file_bytes), dtype_backend="numpy_nullable", skiprows=skip_rows)
            except ImportError:
                pkg = _EXCEL_ENGINE_PACKAGES.get(ext, "openpyxl")
                return None, f"{MISSING_DEPENDENCY_PREFIX}{pkg}"

            # ── Pre-empt the pyarrow mixed-type crash BEFORE it happens ──
            # Any column pandas couldn't confidently type (dtype == 'object')
            # mixes numbers and free text within the same column — extremely
            # common in real-world hand-entered spreadsheets, e.g. a "XEROX"
            # spend column containing mostly numbers (23, 4, 2) plus one cell
            # like "50 pass photo+40 xerox". pl.from_pandas() converts via
            # PyArrow, which infers a single strict Arrow type per column from
            # a sample of values, then hard-crashes with an ArrowInvalid error
            # the instant it hits a value that doesn't fit that inferred type
            # — instead of gracefully falling back to a string column. That
            # crash happens HERE, inside from_pandas(), which is BEFORE the
            # existing object->str "PYARROW FIX" in main.py ever gets a
            # chance to run (that fix only sees already-successfully-loaded
            # data). Stringifying ambiguous columns pre-emptively, right
            # here, sidesteps the crash entirely so mixed text+number columns
            # load instead of taking down the whole file.
            for col in pandas_df.columns:
                if pandas_df[col].dtype == "object":
                    pandas_df[col] = pandas_df[col].apply(lambda v: str(v) if pd.notna(v) else None)

            try:
                df = pl.from_pandas(pandas_df)
            except Exception as exc:
                return None, f"Could not convert Excel data to internal format: {exc}"
        else:
            return None, f"Unsupported file format: .{ext} (accepted: csv, xlsx, xls)"

        if df.is_empty():
            return None, "The file appears to be empty."

        return df, ""

    except Exception as exc:
        return None, f"File read error: {exc}"


def _read_csv_with_encoding_fallback(file_bytes: bytes, skip_rows: int) -> pl.DataFrame:
    """Tries UTF-8 first, then falls back to Latin-1."""
    try:
        return pl.read_csv(
            io.BytesIO(file_bytes),
            infer_schema_length=5000,
            ignore_errors=True,
            null_values=NULL_ALIASES,
            skip_rows=skip_rows, # 👈 Added here
        )
    except Exception:
        return pl.read_csv(
            io.BytesIO(file_bytes),
            infer_schema_length=5000,
            ignore_errors=True,
            null_values=NULL_ALIASES,
            encoding="latin1",
            skip_rows=skip_rows, # 👈 Added here
        )


def preview_raw_rows(file_bytes: bytes, filename: str, n_rows: int = 8) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Reads the first n_rows of a file with NO header inference and NO
    cleaning (header=None) so the user can see exactly what sits on each
    raw row index before choosing how many rows to skip.

    This turns the old "type a number into a box and hope" workflow into
    "look at 'Row 2' in the table, see that it's the real header row, type
    2" — a concrete, verifiable action instead of a guess. Every cell is
    read as a plain string (dtype=str) and blanks are kept as empty strings
    rather than NaN, since this view exists purely for visual inspection,
    not analysis.

    Returns:
        (DataFrame with 'Row N' index / 'Col N' columns, "") on success
        (None, error_message) on failure — including the same
        MISSING_DEPENDENCY_PREFIX signal load_file() uses, so the caller can
        show the same one-click Excel-setup fix instead of a raw traceback.
    """
    try:
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            try:
                raw = pd.read_csv(
                    io.BytesIO(file_bytes), header=None, nrows=n_rows,
                    dtype=str, keep_default_na=False,
                )
            except Exception:
                raw = pd.read_csv(
                    io.BytesIO(file_bytes), header=None, nrows=n_rows,
                    dtype=str, keep_default_na=False, encoding="latin1",
                )
        elif ext in ("xlsx", "xls"):
            try:
                raw = pd.read_excel(io.BytesIO(file_bytes), header=None, nrows=n_rows, dtype=str)
            except ImportError:
                pkg = _EXCEL_ENGINE_PACKAGES.get(ext, "openpyxl")
                return None, f"{MISSING_DEPENDENCY_PREFIX}{pkg}"
        else:
            return None, f"Unsupported file format: .{ext} (accepted: csv, xlsx, xls)"

        if raw is None or raw.empty:
            return None, "The file appears to be empty."

        raw = raw.fillna("")
        raw.index = [f"Row {i}" for i in range(len(raw))]
        raw.columns = [f"Col {i}" for i in range(raw.shape[1])]
        return raw, ""

    except Exception as exc:
        return None, f"Preview error: {exc}"


def ensure_package(package_name: str) -> Tuple[bool, str]:
    """
    Installs a missing optional dependency (e.g. openpyxl for .xlsx support)
    directly from within the running app, so a non-technical user never
    needs to open a terminal or know what "pip" even is — mirrors the same
    one-click, self-healing pattern used for downloading a smaller AI model.

    Thread-safety: guarded by a module-level lock, since this mutates a
    Python environment SHARED by every concurrent user's session in this
    process. Without the lock, two farmers clicking "Set Up Excel Support"
    within the same window could run two pip installs simultaneously,
    which can corrupt each other's install (partial writes to the same
    site-packages entries). The lock just makes the second caller wait for
    the first install to finish rather than racing it.

    Returns:
        (True, "") on success
        (False, error_detail) on failure
    """
    with _pip_install_lock:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Let the current process pick up the newly-installed package
                # without needing a full app restart.
                importlib.invalidate_caches()
                return True, ""
            return False, (result.stderr or result.stdout or "Unknown pip error")[-1000:]
        except Exception as exc:
            return False, str(exc)


def sanitize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalises all column headers to safe snake_case identifiers.
    'Farmer Name' → 'farmer_name_col', 'Group' → 'group_col'

    Guarantees no duplicate column names and 100% SQL safety.
    """
    seen: Dict[str, int] = {}
    clean_map: Dict[str, str] = {}

    for original in df.columns:
        cleaned = (
            str(original)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("&", "and")
            .replace("%", "pct")
            .replace("#", "no")
        )

        # ---------------------------------------------------------
        # CATCH-ALL: the hand-picked replacements above only cover the
        # characters we anticipated. Real spreadsheets throw plenty of
        # others at us — e.g. pandas names a blank header "Unnamed: 2",
        # and that colon was slipping straight through into the SQL
        # identifier ("unnamed:_2_col"), which DuckDB then rejects with a
        # genuine parser error the instant the AI references that column
        # in a query ("syntax error at or near ':'") — breaking any chart
        # that touches it. Rather than extending the replace-list every
        # time a new punctuation mark turns up, strip out ANYTHING that
        # isn't a-z, 0-9, or underscore, unconditionally. This guarantees
        # a valid SQL identifier no matter what a farmer's original column
        # header looked like.
        # ---------------------------------------------------------
        cleaned = re.sub(r"[^a-z0-9_]", "_", cleaned)

        # Clean up multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned).strip("_") or "col"

        # SQL identifiers can't start with a digit unqualified (DuckDB will
        # choke on it just like the colon case above) — e.g. a "2024_sales"
        # header. Prefix with a letter so it's always safe unquoted.
        if cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"

        # ---------------------------------------------------------
        # THE FIX: Append '_col' to make it 100% safe from SQL keywords,
        # but ONLY if it doesn't already end in '_col' (prevents _col_col)
        # ---------------------------------------------------------
        if not cleaned.endswith("_col"):
            cleaned = f"{cleaned}_col"

        # Deduplicate
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0

        clean_map[original] = cleaned

    return df.rename(clean_map)


# ─────────────────────────────────────────────────────────
# 2. HEALTH PROFILING
# ─────────────────────────────────────────────────────────

def generate_health_report(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Produces a comprehensive data-quality audit covering:
      - Null counts per column
      - Duplicate rows
      - Type mismatches (text columns with mostly numeric content)
      - Statistical outliers (z-score and IQR)
      - Overall health score (0–100)
    """
    if df is None or df.is_empty():
        return {
            "total_rows": 0, "total_cols": 0, "schema": {},
            "missing": {}, "duplicate_rows": 0,
            "type_mismatches": [], "outliers": {}, "health_score": 0,
        }

    total = df.height
    schema: Dict[str, str] = {col: str(dt) for col, dt in zip(df.columns, df.dtypes)}
    missing: Dict[str, Dict] = {}
    type_mismatches: List[Dict] = []
    outliers: Dict[str, Dict] = {}

    for col in df.columns:
        null_n = df[col].null_count()
        if null_n > 0:
            missing[col] = {"count": null_n, "pct": round(null_n / total * 100, 2)}

        dtype = df[col].dtype

        # ── Type-mismatch detection ──────────────────────
        if dtype in STRING_DTYPES:
            s = df[col].drop_nulls().cast(pl.Utf8)
            if len(s) > 0:
                ratio = s.str.contains(r"^-?\d+(\.\d+)?$").mean()
                if ratio is not None and 0.3 < float(ratio) < 1.0:
                    type_mismatches.append({
                        "column": col,
                        "numeric_ratio": round(float(ratio), 2),
                        "suggestion": "Consider casting to Int64 or Float64",
                    })

        # ── Outlier detection ────────────────────────────
        if dtype in NUMERIC_DTYPES:
            s = df[col].drop_nulls()
            if len(s) > 4:
                mean_v = float(s.mean())
                std_v = float(s.std() or 0)
                if std_v > 0:
                    z_count = int(((s - mean_v) / std_v).abs().gt(3).sum())
                    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    iqr = q3 - q1
                    iqr_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                    if z_count > 0:
                        outliers[col] = {
                            "z_score_count": z_count,
                            "iqr_count": iqr_count,
                        }

    dup_count = int(df.is_duplicated().sum())
    issues = len(missing) + (1 if dup_count > 0 else 0) + len(outliers) + len(type_mismatches)
    health_score = max(0, 100 - issues * 5)

    return {
        "total_rows": total,
        "total_cols": df.width,
        "schema": schema,
        "missing": missing,
        "duplicate_rows": dup_count,
        "type_mismatches": type_mismatches,
        "outliers": outliers,
        "health_score": health_score,
    }


# ─────────────────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────────────────

def auto_fill_missing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Automatically fills empty/null values with 0.
    Only triggered if the user checks the auto-fill box in the UI.
    """
    # Safely fills nulls with 0 without overwriting manual string/numeric edits
    return df.fill_null(0).fill_nan(0)

# ─────────────────────────────────────────────────────────
# 4. DUCKDB LAYER
# ─────────────────────────────────────────────────────────

def create_duckdb_connection(
    df: pl.DataFrame,
) -> Tuple[Optional[duckdb.DuckDBPyConnection], str]:
    """
    Creates an in-memory DuckDB connection and *materialises* the DataFrame
    as a permanent table named `dataset`.

    Using CREATE TABLE (rather than register) avoids any garbage-collection
    issues with the pandas/polars reference inside a Streamlit session.

    Each Streamlit session stores its own connection in st.session_state, so
    concurrent farmers each get an independent in-memory DuckDB instance —
    no cross-user data ever shares a connection. The only cost to watch on a
    shared server is aggregate memory: N concurrent sessions each hold their
    own copy of their (cleaned) dataset in memory at once.
    """
    try:
        pandas_df = df.to_pandas()
        conn = duckdb.connect(":memory:")
        # Register as a temporary view, then bake into a real table
        conn.register("_staging", pandas_df)
        conn.execute("CREATE TABLE dataset AS SELECT * FROM _staging")
        conn.unregister("_staging")
        return conn, ""
    except Exception as exc:
        return None, f"DuckDB init error: {exc}"


def execute_sql_query(
    conn: duckdb.DuckDBPyConnection, sql: str
) -> Tuple[Optional[pl.DataFrame], str]:
    """
    Executes an AI-generated SQL query with security guards.

    Security rules:
      1. Query must start with SELECT or WITH (CTE).
      2. Blocked DML/DDL keywords are rejected immediately.
      3. Results are capped at 500 rows to prevent memory issues.

    Returns:
        (Polars DataFrame, "") on success
        (None, error_message) on failure
    """
    stripped = sql.strip().lstrip("(").upper()

    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        return None, "Security: Only SELECT/WITH queries are permitted."

    if _SQL_BLOCKLIST.search(sql):
        match = _SQL_BLOCKLIST.search(sql)
        blocked_kw = match.group(0) if match else "unknown"
        return None, f"Security: The keyword '{blocked_kw}' is not allowed."

    # Ensure there is always a row limit to prevent runaway queries
    safe_sql = _inject_limit(sql, limit=500)

    try:
        result = conn.execute(safe_sql).pl()
        return result, ""
    except Exception as exc:
        return None, str(exc)


def _inject_limit(sql: str, limit: int = 500) -> str:
    """Appends a LIMIT clause if the query does not already have one."""
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip(";").strip() + f" LIMIT {limit}"


def get_schema_description(df: pl.DataFrame) -> str:
    """
    Builds a rich, LLM-friendly schema description with:
    - Exact column names (as stored in the `dataset` table)
    - Data types
    - Unique value counts
    - Sample values for categorical columns (≤ 30 unique)
    - Min / max for numeric columns
    - Separate sections for grouping vs. metric columns
    """
    grouping_cols: List[str] = []
    metric_cols: List[str] = []
    id_cols: List[str] = []

    for col in df.columns:
        n_unique = df[col].n_unique()
        dtype = df[col].dtype
        if dtype in NUMERIC_DTYPES:
            if n_unique >= df.height * 0.9:
                id_cols.append(col)
            else:
                metric_cols.append(col)
        else:
            grouping_cols.append(col)

    lines: List[str] = [
        "╔══ DATABASE SCHEMA ══╗",
        f"  TABLE NAME : `dataset`  ← use this EXACT name in every SQL query",
        f"  TOTAL ROWS : {df.height:,}",
        f"  TOTAL COLS : {df.width}",
        "",
    ]

    if grouping_cols:
        lines.append("── GROUPING / CATEGORICAL COLUMNS (use in GROUP BY, WHERE, ILIKE) ──")
        for col in grouping_cols:
            n_unique = df[col].n_unique()
            samples = _get_samples(df, col)
            lines.append(f"  • {col}  ({str(df[col].dtype)}, {n_unique} unique){samples}")

    if metric_cols:
        lines.append("")
        lines.append("── METRIC / NUMERIC COLUMNS (use in COUNT, SUM, AVG, MIN, MAX) ──")
        for col in metric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            lines.append(f"  • {col}  ({str(df[col].dtype)}, range: {col_min} → {col_max})")

    if id_cols:
        lines.append("")
        lines.append("── ID / HIGH-CARDINALITY COLUMNS (avoid in GROUP BY) ──")
        for col in id_cols:
            lines.append(f"  • {col}  ({str(df[col].dtype)}, {df[col].n_unique()} unique)")

    lines.append("╚══════════════════════╝")
    return "\n".join(lines)


def _get_samples(df: pl.DataFrame, col: str, n: int = 5) -> str:
    n_unique = df[col].n_unique()
    if n_unique > 30:
        return f"  [{n_unique} distinct values]"
    samples = df[col].drop_nulls().unique().head(n).cast(pl.Utf8).to_list()
    return f"  | samples: {', '.join(samples)}"
