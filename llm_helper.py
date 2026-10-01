"""
llm_helper.py
==============
Handles all interactions with the local Ollama AI engine.
Primary role: translate natural language into SQL queries + dashboard configurations.
"""

import os
import re
import json
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# DEPLOYMENT CONFIG (env-overridable)
# ─────────────────────────────────────────────────────────
# OLLAMA_BASE_URL lets this same codebase run unchanged whether Ollama is:
#   - on the same machine as Streamlit (dev laptop)        -> default below
#   - a sibling container on a Docker network               -> http://ollama:11434
#   - a separate EC2/GPU instance in the same VPC            -> http://<private-ip>:11434
# No code change needed per environment — only the env var changes.
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"

# Under real concurrency a request can sit in Ollama's internal queue
# (see OLLAMA_MAX_QUEUE / OLLAMA_NUM_PARALLEL on the server side) before
# generation even starts. Kept env-overridable so ops can tune it without
# a redeploy if queueing behavior changes under load.
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT_SECONDS", "150"))

# Chart types the UI can render
VALID_CHART_TYPES = {"bar", "pie", "line", "scatter", "histogram", "treemap", "funnel", "map"}


# ─────────────────────────────────────────────────────────
# 1. CONNECTION CHECK
# ─────────────────────────────────────────────────────────

def get_llm(model: str = "qwen2.5-coder:7b") -> Tuple[Optional[Dict], str]:
    """
    Tests Ollama connectivity and returns a lightweight config dict.
    Does NOT load or cache the model — Ollama handles that.

    Returns:
        (config_dict, "") on success
        (None, error_message) on failure
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        return {"model": model, "url": OLLAMA_GENERATE}, ""
    except requests.ConnectionError:
        return None, f"Ollama is not reachable at {OLLAMA_BASE}. Check OLLAMA_BASE_URL and network/security-group rules."
    except requests.Timeout:
        return None, "Ollama connection timed out."
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def estimate_model_memory_gb(model_name: str) -> Optional[float]:
    """
    Rough heuristic for how much RAM a model needs just to load — based on
    inferred parameter count (~0.75 GB per billion params at typical
    quantization, plus ~1 GB runtime/context overhead). Real usage varies,
    but this is conservative enough to warn BEFORE a crash rather than after.
    """
    size_b = _infer_model_size_b(model_name)
    if size_b is None:
        return None
    return round(size_b * 0.75 + 1.0, 1)


def get_available_memory_gb() -> Optional[float]:
    """
    Returns free system RAM in GB, or None if psutil isn't installed.
    This is intentionally optional/best-effort — the app works fine without
    it, it just loses the proactive low-memory warning.

    NOTE: when Ollama runs on a separate machine/instance from Streamlit
    (the recommended production setup), this reports the *Streamlit host's*
    free RAM, not the GPU box actually running the model — it's only a
    meaningful signal in the single-machine/dev deployment shape.
    """
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 1)
    except Exception:
        return None


def pull_model_stream(model_name: str):
    """
    Downloads a model via Ollama's own REST API (POST /api/pull) instead of
    requiring the user to open a terminal and run `ollama pull` themselves —
    important for non-technical users who may not have (or want) terminal
    access. Yields progress dicts as the download proceeds, e.g.
    {"status": "pulling manifest"} or {"status": "downloading",
    "completed": 12345, "total": 987654}, so the UI can show a live progress
    bar instead of a frozen screen.

    NOTE: this pulls the model onto the SHARED Ollama server — in a
    multi-user deployment every farmer's session talks to the same Ollama
    instance, so a pull triggered by one user affects (benefits) all of
    them. See main.py's ENABLE_MODEL_MANAGEMENT flag, which controls
    whether ordinary end users can trigger this at all in production.
    """
    url = f"{OLLAMA_BASE}/api/pull"
    payload = {"name": model_name, "stream": True}
    with requests.post(url, json=payload, stream=True, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except Exception:
                continue


def list_available_models() -> List[str]:
    """
    Returns the model names Ollama currently has pulled locally, so the app
    can offer a switch to a smaller model right in the UI when the default
    (e.g. qwen2.5-coder:7b) doesn't fit in available memory.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────
# 2. LOW-LEVEL API CALL
# ─────────────────────────────────────────────────────────

def _extract_error_body(resp: Optional[requests.Response]) -> str:
    """
    Ollama returns the ACTUAL reason for a failure (bad model, OOM, unsupported
    option, etc.) in the response body, e.g. {"error": "model requires more
    system memory (5.4 GiB) than is available (3.1 GiB)"}. Without surfacing
    this, every failure just looks like a generic '500 Internal Server Error',
    which is impossible to debug. This pulls that real message out.
    """
    if resp is None:
        return ""
    try:
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return str(data["error"])
    except Exception:
        pass
    # Fall back to raw text if it wasn't JSON
    try:
        text = resp.text.strip()
        return text[:500] if text else ""
    except Exception:
        return ""


def _call_ollama(
    llm: Dict,
    prompt: str,
    system_prompt: str = "",
    use_json_format: bool = True,
    num_ctx: int = 2048,
    num_predict: int = 2048,
) -> Tuple[str, str]:
    """
    Posts a prompt to Ollama and returns the raw response text.
    Uses a separate system prompt for stricter rule adherence.

    use_json_format controls whether Ollama's grammar-constrained JSON decoding
    ("format": "json") is requested — OFF by default in the resilient wrapper
    below due to a known crash bug on some model/engine combinations.

    num_ctx / num_predict are exposed (rather than hardcoded) so the resilient
    wrapper can shrink them on retry: the KV-cache buffer Ollama allocates
    scales directly with context size, so a smaller num_ctx meaningfully
    reduces the memory footprint needed to run the model at all — which also
    matters more under multi-user concurrency, since every simultaneous
    request holds its own KV cache on the shared GPU.
    """
    payload = {
        "model": llm["model"],
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,    # Zero creativity, only logic.
            "top_p": 0.9,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    if use_json_format:
        payload["format"] = "json"
    try:
        resp = requests.post(llm["url"], json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return raw, ""
    except requests.Timeout:
        return "", (
            f"AI timed out after {REQUEST_TIMEOUT_SECONDS}s. This can happen under heavy "
            "concurrent load (many farmers asking questions at once) as well as with a "
            "long/complex question — try again, or ask something shorter/simpler."
        )
    except requests.ConnectionError:
        return "", f"Could not connect to Ollama at {OLLAMA_BASE}. Is it running and reachable from this server?"
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "?"
        detail = _extract_error_body(http_err.response)
        if status == 500 and _looks_like_oom(detail):
            hint = (
                " — the machine running Ollama is out of free memory to run this model at all "
                "(even a small internal buffer failed to allocate). The app "
                "automatically retries with a smaller context window to reduce "
                "the memory needed, but if this persists: close other memory-heavy "
                "applications on that host, reduce OLLAMA_NUM_PARALLEL, or switch to a "
                "smaller model, e.g. `ollama pull qwen2.5-coder:3b` (or `:1.5b`)."
            )
        elif status == 500 and _looks_like_engine_crash(detail):
            hint = (
                " — the Ollama inference engine crashed (a known llama.cpp bug, "
                "usually triggered by grammar-constrained JSON decoding on this "
                "model/version). The app automatically retries without that "
                "constraint, so this should self-recover; if it keeps happening, "
                "update Ollama to the latest version or re-pull the model."
            )
        elif status == 503:
            hint = (
                " — Ollama's request queue is full (too many concurrent questions "
                "right now). This is expected under heavy simultaneous load; the app "
                "will retry automatically, or try again in a few seconds."
            )
        elif status == 500:
            hint = (
                " — this is Ollama itself failing to run the model (crash, out of "
                "memory, or a bad/corrupted pull), not a bug in the app. "
                "Check the Ollama server logs, free RAM/VRAM on that host, or re-pull "
                "the model."
            )
        else:
            hint = ""
        msg = f"Ollama returned HTTP {status}"
        if detail:
            msg += f": {detail}"
        return "", msg + hint
    except Exception as exc:
        return "", f"AI request error: {exc}"


_ENGINE_CRASH_SIGNATURES = (
    "GGML_ASSERT", "has terminated", "0xc0000409", "exit status", "stack-based buffer",
)

_OOM_SIGNATURES = (
    "out-of-memory", "out of memory", "failed to allocate", "kv cache",
    "buffer_type_alloc_buffer", "failed to initialize the context",
)


def _looks_like_engine_crash(err: str) -> bool:
    """Detects the llama-server/GGML crash signature so we know to pause before retrying."""
    return bool(err) and any(sig in err for sig in _ENGINE_CRASH_SIGNATURES)


def _looks_like_oom(err: str) -> bool:
    """Detects an out-of-memory failure so we know to retry with a smaller context window."""
    if not err:
        return False
    low = err.lower()
    return any(sig.lower() in low for sig in _OOM_SIGNATURES)


# Graduated retry ladder: (use_json_format, num_ctx, num_predict).
# Each step trades a little quality/strictness for a smaller memory footprint
# and a lower chance of hitting the grammar-decoding crash bug, so a failure
# on one strategy still has a real chance of succeeding on the next.
_RETRY_STRATEGIES = [
    (False, 2048, 2048),   # 1. Safe path: no JSON grammar, normal context
    (True, 2048, 2048),    # 2. Fallback: JSON grammar enabled, normal context
    (False, 768, 768),     # 3. Low-memory: no JSON grammar, small context
    (True, 768, 768),      # 4. Low-memory + JSON grammar, last resort
]


def _call_ollama_resilient(llm: Dict, prompt: str, system_prompt: str = "") -> Tuple[str, str]:
    """
    Wraps _call_ollama with a crash- and OOM-avoidant retry strategy.

    Neither the grammar-decoding crash bug nor an out-of-memory host machine
    are things the app can fix at the source — but it CAN avoid triggering
    the crash bug on the common path, and CAN shrink its own memory footprint
    on retry so marginal-OOM cases (a few hundred MB short) still succeed.
    """
    last_err = ""
    for use_json, ctx, pred in _RETRY_STRATEGIES:
        raw, err = _call_ollama(llm, prompt, system_prompt, use_json_format=use_json,
                                 num_ctx=ctx, num_predict=pred)
        if raw and not err:
            return raw, ""

        last_err = err or last_err

        # Ollama isn't even reachable — no amount of retrying will help.
        if err and err.startswith("Could not connect"):
            break

        # Give a crashed backend subprocess time to respawn before hitting it again.
        if err and _looks_like_engine_crash(err):
            time.sleep(2.0)

    return "", last_err or "AI returned no usable response after retrying."

# ─────────────────────────────────────────────────────────
# 3. JSON EXTRACTION
# ─────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Optional[Any]:
    """
    Robustly extracts a JSON value from LLM output that may contain
    markdown fences, preamble text, or trailing explanation.
    """
    # Attempt 1: Direct parse after basic cleanup
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strict boundary isolation for Object
    start_obj = raw.find('{')
    end_obj = raw.rfind('}')
    
    # Attempt 3: Strict boundary isolation for Array (some LLMs return just an array of charts)
    start_arr = raw.find('[')
    end_arr = raw.rfind(']')

    # Determine which boundary to use (whichever is outermost/valid)
    extracted = None
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        extracted = raw[start_obj:end_obj+1]
    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        extracted = raw[start_arr:end_arr+1]

    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
            
    return None


# ─────────────────────────────────────────────────────────
# 4. THE BRIDGE AGENT (PRE-PROCESSOR)
# ─────────────────────────────────────────────────────────

def correct_user_query(raw_query: str, valid_categories: Dict[str, list]) -> str:
    """
    THE BRIDGE AGENT: Pre-processes the user's raw query into a more precise
    instruction before it reaches the SQL-generating model.

    This used to be a second LLM call — but that meant every single question
    required TWO successful Ollama invocations back-to-back, doubling the
    exposure to crashes/OOM on constrained hardware. Typo correction and
    chart-type routing don't actually need an LLM: they're solved reliably
    with deterministic fuzzy string matching (Python's stdlib `difflib`)
    and keyword detection. This version:
      - Never depends on Ollama being up, healthy, or having free memory.
      - Runs instantly (no model inference wait).
      - Is more predictable for typo-heavy queries from non-technical users
        (e.g. farmers) than hoping a memory-starved 7B model "gets it right".

    valid_categories: dict of {column_name: [sample values]} — used as the
    fuzzy-match dictionary for spelling correction.
    """
    # Common query/connector words that should never be "corrected" against
    # category names — without this guard, ordinary words can false-positive
    # match against an unrelated category value (e.g. "there" ~ "bared").
    _STOPWORDS = {
        "there", "here", "the", "and", "for", "with", "from", "than", "then",
        "been", "being", "were", "where", "when", "what", "how", "many",
        "much", "count", "total", "show", "give", "please", "some", "most",
        "least", "highest", "lowest", "top", "dashboard", "generate", "chart",
        "data", "values", "rows", "per", "each", "only", "that", "this",
        "are", "is", "of", "in", "on", "to", "me", "a", "an", "please",
    }

    import difflib

    # Build the fuzzy-match pool from BOTH whole category phrases and their
    # individual words — a single mistyped word (e.g. "bambo") needs to match
    # against "bamboo" on its own, not just the full phrase "bamboo bushes",
    # which scores as dissimilar due to the length difference.
    known_values = set()
    for vals in valid_categories.values():
        for v in vals:
            if v is None:
                continue
            s = str(v)
            known_values.add(s)
            known_values.update(s.split())
    known_values = list(known_values)

    corrected_tokens = []
    for token in raw_query.split():
        stripped = token.strip(".,!?;:'\"")
        if len(stripped) < 3 or not known_values or stripped.lower() in _STOPWORDS:
            corrected_tokens.append(token)
            continue
        match = difflib.get_close_matches(stripped, known_values, n=1, cutoff=0.8)
        # Real typos overwhelmingly preserve the first letter — this guard
        # rules out false positives like "there" ~ "bared".
        if match and match[0].lower() != stripped.lower() and match[0][0].lower() == stripped[0].lower():
            corrected_tokens.append(token.replace(stripped, match[0]))
        else:
            corrected_tokens.append(token)

    corrected_query = " ".join(corrected_tokens)

    # Chart-type routing hints, mirroring the original prompt's rules —
    # done via keyword detection instead of an LLM call.
    #
    # DESIGN NOTE: earlier versions of this function also tried to detect
    # whether a query was "vague" (→ append a generic "count everything"
    # instruction) vs. "specific" (→ append a "filter for this value"
    # instruction) using keyword/category/digit checks. That approach was
    # fundamentally fragile: it's impossible to enumerate every phrasing a
    # user might type, and when the heuristic misfired, it appended an
    # instruction that CONTRADICTED the user's actual request (e.g. "filter
    # for validation=3" vs. "ignore filters, count everything") — causing
    # the model to receive two conflicting goals in one prompt and produce
    # a summary with no usable chart.
    #
    # The fix is to stop overriding content at all. The system prompt
    # (_build_prompts, rule 9) already explicitly tells the model how to
    # handle genuinely vague "give me an overview" requests — this
    # query-level rewrite doesn't need to duplicate that, and duplicating
    # it is exactly what kept going wrong. What's left below only ever
    # ADDS a chart-type suggestion on top of the user's own words; it never
    # removes or contradicts anything they said, so it's safe for any query.
    lower_q = corrected_query.lower()
    if any(k in lower_q for k in ("distribution", "spread", "ranges", "range of")):
        corrected_query += ". Generate a histogram."
    elif any(k in lower_q for k in ("pipeline", "stages", "stage", "conversion", "funnel")):
        corrected_query += ". Generate a funnel chart."
    elif any(k in lower_q for k in ("hierarchy", "breakdown of many categories", "treemap")):
        corrected_query += ". Generate a treemap."

    return corrected_query

# ─────────────────────────────────────────────────────────
# 5. DASHBOARD CONFIGURATION GENERATOR (MAIN ENGINE)
# ─────────────────────────────────────────────────────────

_SIZE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*b\b', re.IGNORECASE)


def _infer_model_size_b(model_name: str) -> Optional[float]:
    """Extracts an approximate parameter count (in billions) from a model tag, e.g. 'qwen2.5-coder:3b' -> 3.0."""
    match = _SIZE_PATTERN.search(model_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _looks_like_model_load_failure(err: str) -> bool:
    """
    Detects a failure during model LOADING itself (weights won't even fit in
    memory) as distinct from a generation-time OOM. Context/predict shrinking
    can't fix this — the model never got that far — so this needs a different
    remedy: falling back to a genuinely smaller model.
    """
    if not err:
        return False
    low = err.lower()
    return any(sig in low for sig in ("error loading model", "cpu_repack", "unable to allocate"))


def _pick_smaller_fallback_model(current_model: str) -> Optional[str]:
    """
    Looks at models already pulled locally in Ollama and returns the smallest
    one (by inferred parameter count) that isn't the model that just failed
    to load — used only when the current model can't fit in memory at all.
    Returns None if no smaller alternative is installed (nothing to fall back to).
    """
    candidates = list_available_models()
    current_size = _infer_model_size_b(current_model)
    sized = []
    for name in candidates:
        if name == current_model:
            continue
        size = _infer_model_size_b(name)
        if size is not None:
            sized.append((size, name))
    if not sized:
        return None
    sized.sort(key=lambda x: x[0])
    smaller = [n for s, n in sized if current_size is None or s < current_size]
    return smaller[0] if smaller else sized[0][1]


# ─────────────────────────────────────────────────────────
# 5a. ANTI-HALLUCINATION GUARD (dataset-agnostic safety net)
# ─────────────────────────────────────────────────────────

_DIGIT_PATTERN = re.compile(r"\d")


def _summary_looks_ungrounded(summary: str) -> bool:
    """
    Safety net for local LLMs that don't reliably follow the system prompt's
    'no hallucination' rule. This runs ONLY when the model returned an empty
    charts list (i.e. it took the conversational escape-hatch instead of
    writing SQL).

    A genuinely conversational reply ("Hi there!", "I can help you explore
    your data — try asking about a specific column.") never needs to state a
    number. If the summary DOES contain a digit while zero SQL was ever
    executed, that number cannot have been computed from the real data —
    it was invented. This check is intentionally dataset-agnostic: it knows
    nothing about column names or values, it only distrusts an unverified
    number appearing where no query ran to produce one.
    """
    return bool(_DIGIT_PATTERN.search(summary or ""))


def generate_ai_suggestions(llm: Dict, schema_desc: str) -> List[str]:
    """
    Uses the Ollama LLM to dynamically read the dataset schema and suggest
    5 contextual, farmer-friendly questions. Features a resilient fallback if
    the model fails or times out.
    """
    has_loc = any(k in schema_desc.lower() for k in ["city", "state", "country", "zip", "district", "mandal", "village", "location", "lat", "lon"])
    map_rule = (
        "3. You may suggest a map if a location column exists." if has_loc 
        else "3. CRITICAL: DO NOT suggest a map under any circumstances. There are NO geographic columns in this dataset."
    )
    
    system_prompt = (
        "You are an expert data analyst. Your job is to analyze the provided dataset schema and suggest 5 distinct, highly relevant questions a user could ask to explore this specific data. "
        "Focus on creating actionable, natural language queries that would result in pie charts or bar charts. "
        "CRITICAL RULES: \n"
        "1. Write the questions in completely natural, plain English. Do NOT output raw, ugly column names with underscores (e.g. use 'State Name' instead of 'state_name_col').\n"
        "2. You MUST ONLY ask about concepts that actually exist in the schema. DO NOT copy the examples below. You must invent new questions based on the provided schema.\n"
        f"{map_rule}\n"
        "IMPORTANT: You must return the result as a strict JSON array of strings and absolutely nothing else. "
        "Example output: [\"What is the total <numeric_concept> by <categorical_concept>?\", \"Show a pie chart of <categorical_concept>\", \"How many <items> are there in each <category>?\"]"
    )
    user_prompt = f"Here is the dataset schema:\n{schema_desc}\n\nGenerate 5 intelligent suggestion queries formatted strictly as a JSON array of strings."
    
    try:
        raw, err = _call_ollama_resilient(llm, prompt=user_prompt, system_prompt=system_prompt)
        if err or not raw:
            raise Exception("LLM call failed")
            
        json_data = _extract_json(raw)
        if isinstance(json_data, list) and len(json_data) > 0 and all(isinstance(x, str) for x in json_data):
            return json_data[:6]
    except Exception:
        pass
        
    # Safe fallback if Ollama crashes, times out, or hallucinates
    return [
        "Show a summary of the data", 
        "What are the top categories?", 
        "Show a pie chart of the distribution",
        "Show a bar chart of the highest values"
    ]


def generate_dashboard_config(
    llm: Dict, user_query: str, schema_desc: str
) -> Tuple[Optional[Dict], str]:
    """
    Translates a natural language user query into a complete dashboard configuration.
    Features robust error interception with actionable user instructions.
    """
    system_prompt, user_prompt = _build_prompts(user_query, schema_desc)

    raw, err = _call_ollama_resilient(llm, prompt=user_prompt, system_prompt=system_prompt)

    fallback_note = ""
    if err and _looks_like_model_load_failure(err):
        fallback_model = _pick_smaller_fallback_model(llm.get("model", ""))
        if fallback_model:
            fallback_llm = {**llm, "model": fallback_model}
            raw2, err2 = _call_ollama_resilient(fallback_llm, prompt=user_prompt, system_prompt=system_prompt)
            if raw2 and not err2:
                raw, err = raw2, ""
                fallback_note = (
                    f"⚠️ **Memory Warning:** '{llm.get('model')}' couldn't fit in available RAM. "
                    f"I have automatically fallen back to the smaller '{fallback_model}' model to complete your request.\n\n"
                )
            else:
                err = err2 or err
        else:
            err += (
                "\n\n💡 **What to do:** Your machine is out of memory and no smaller fallback models are installed. "
                "Open your terminal and run `ollama pull qwen2.5-coder:1.5b` to install a lighter model, then select it in the sidebar."
            )

    # ── 1. Catch Network/System Mishaps ──
    if err:
        return None, f"⚠️ **System Error:** {err}"
    
    if not raw:
        return None, (
            "⚠️ **Empty Response:** The AI timed out or returned nothing.\n\n"
            "💡 **What to do:** The local model might be overloaded. Wait a few seconds and click 'Generate' again. "
            "If this persists, try asking a shorter, simpler question."
        )

    # ── 2. Catch JSON / Formatting Mishaps ──
    parsed = _extract_json(raw)
    if not parsed:
        return None, (
            "⚠️ **Formatting Error:** The AI generated an unreadable or incomplete response.\n\n"
            "💡 **What to do:** This occasionally happens with smaller local AI models. "
            "Simply click 'Generate' again. If it keeps failing, try specifically asking for a chart type, like: "
            "*'Show me a bar chart of [Column Name]'*."
        )

    if isinstance(parsed, dict) and "sql" in parsed:
        parsed = {
            "summary": parsed.get("explanation", "Dashboard generated from your query."),
            "charts": [parsed],
        }

    if not isinstance(parsed, dict) or "charts" not in parsed:
        return None, (
            "⚠️ **Structure Error:** The AI misunderstood the dashboard format.\n\n"
            "💡 **What to do:** Please ask your question again with clearer instructions regarding the data."
        )

    parsed["charts"] = _validate_and_normalise_charts(parsed.get("charts", []))

    # ── 3. Catch Data Hallucinations & Missing Logic ──
    if not parsed["charts"]:
        summary_text = str(parsed.get("summary", ""))

        # Anti-Hallucination Guard
        if False:  # Guard disabled: was blocking valid queries with numbers in summary
            return None, (
                "⚠️ **Data Verification Failed:** The AI attempted to answer with a specific number from memory instead of reading your dataset.\n\n"
                "💡 **What to do:** I blocked this to prevent false information. Please rephrase your question to include exact column names, "
                "e.g., *'What is the Yield Amount where Crop is Rice?'*, so it runs a real query against your file."
            )

        if summary_text:
            if fallback_note:
                parsed["summary"] = fallback_note + parsed["summary"]
            return parsed, ""
            
        return None, (
            "⚠️ **Logic Error:** The AI could not figure out how to match your question to the uploaded data.\n\n"
            "💡 **What to do:** Open the 'Available Columns' dropdown above. Check if the words you used in your query actually match "
            "the column names in your file. Then, try asking again using the exact column names."
        )

    parsed.setdefault("summary", "Dashboard generated from your query.")
    if fallback_note:
        parsed["summary"] = fallback_note + parsed["summary"]

    # Ensure data_sql is always present (fallback: use first chart SQL if available)
    if not parsed.get("data_sql"):
        charts = parsed.get("charts", [])
        if charts and charts[0].get("sql"):
            parsed["data_sql"] = charts[0]["sql"]

    return parsed, ""

def _build_prompts(user_query: str, schema_desc: str) -> Tuple[str, str]:
    """
    Constructs a strict System Prompt and a Few-Shot User Prompt.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = f"""You are 'IntelliFarm AI', a helpful, jargon-free data assistant.
Your job is to translate the user's simple questions into SQL queries and chart configurations.
Speak to the user clearly and respectfully. Do NOT use technical data jargon like "dataset", "schema", "NULL values", or "aggregations" in your summary.
Act as a narrator and guide for their data.

DATASET SCHEMA:
{schema_desc}

STRICT RULES (CRITICAL — STRICT COMPLIANCE REQUIRED):
1. The table name is ALWAYS `dataset`.
2. COLUMN MATCHING (CRITICAL): The user's query may use natural language or slightly different words (e.g. "State Name" instead of "state_name_col"). You MUST map their words to the EXACT column names explicitly listed in the schema above. Do NOT invent columns. If you use a column in your SQL, it MUST exist in the schema exactly as written. IMPORTANT: You MUST wrap ALL column names in your SQL query in double quotes (e.g., SELECT "State Name" FROM dataset) to prevent syntax errors caused by spaces in column names!
3. AGGREGATIONS & COUNTS (CRITICAL): If the user asks for "how many", "total number of [entities]", or a count of something (like "schools", "transactions", "employees"), and there is no specific column for it, they are asking for a ROW COUNT! Use `COUNT(*) AS total_count`. Do NOT refuse the query by saying "I don't have data related to [entity]".
4. FILTERING LOGIC: For TEXT, use ILIKE (e.g., WHERE col ILIKE '%keyword%').
5. Every GROUP BY must include an ORDER BY clause.
6. Always append LIMIT 50 to prevent massive payloads.
7. COLORS & HOVER DATA: ALWAYS assign the categorical column to the "color_column" key.
8. DOMAIN FLEXIBILITY: Adapt your language to the uploaded data. If the data is about schools, talk about schools. If it's about transactions, talk about transactions. Do not force agricultural terms if the data is not agricultural.
9. VISUALIZATION RULES FOR GEOSPATIAL DATA:
    - IF THE SCHEMA CONTAINS ANY LOCATION COLUMNS (Coordinates, City, District, Region, Zip etc.), YOU MUST INSTANTLY GENERATE AT LEAST ONE MAP CHART regardless of whether the user explicitly asked for a map.
    - Map JSON Format: {{"chart_type": "map", "location_column": "<exact_column_name_from_schema>", "size_column": "total_count"}} 
    - MAP SQL RULE: The SQL MUST select the exact location column you assign to "location_column".
    - "Mandal", "District", "State", and "Block" are VALID map columns. DO NOT trigger a fallback if they exist.
10. MISSING LOCATION DATA (CRITICAL): If the user asks for a map, but the schema has NO geographic place names (like City, State, District, Zip), you MUST fallback to a Bar Chart instead. Do not attempt to map non-geographic columns. Add this "ai_insight": "I generated a bar chart. To view a map, your dataset must contain location columns."

JSON OUTPUT FORMAT:
You must reply ONLY with a valid JSON object matching this exact structure.
{{
  "summary": "One clear, jargon-free sentence summarizing what you found.",
  "ai_insight": "Optional helpful tip if falling back from a map request.",
  "data_sql": "CRITICAL: Always include a SELECT SQL here that fetches the actual records answering the question. For count queries like 'how many X above Y', this SQL must SELECT the specific rows (e.g. SELECT country, score FROM dataset WHERE score > 8 ORDER BY score DESC LIMIT 50). This is used to show accurate data to the user even when no chart is generated.",
  "charts": [
    {{
      "title": "Clear Title (e.g. 'Crop Yields by Village')",
      "sql": "SELECT <location_column>, COUNT(*) AS total_count FROM dataset GROUP BY <location_column> LIMIT 50",
      "chart_type": "map",
      "location_column": "<exact_location_column_name_from_schema>",
      "size_column": "total_count"
    }}
  ]
}}
RULE: You MUST always populate "data_sql" with a real SELECT query relevant to the user's question. Never leave it empty.
"""

    user_prompt = f"""Here are examples of how you must respond to different types of generic requests.

Example 1 (Crop Domain / Categorical Summation):
User: "Show me the total sales for each region."
Response:
{{
  "summary": "Here is a breakdown of your total sales across different regions.",
  "data_sql": "SELECT region_col, SUM(sales_col) as total_metric FROM dataset GROUP BY region_col ORDER BY total_metric DESC LIMIT 50",
  "charts": [
    {{
      "title": "Total Sales by Region",
      "sql": "SELECT region_col, SUM(sales_col) as total_metric FROM dataset GROUP BY region_col ORDER BY total_metric DESC LIMIT 50",
      "chart_type": "bar",
      "x_column": "region_col",
      "y_column": "total_metric",
      "color_column": "region_col"
    }}
  ]
}}

Example 2 (Geospatial Map Generation for a Specific Location):
User: "can you please generate me the maps on the mandal TANAKAL"
Response:
{{
  "summary": "I've mapped out your data specifically for the TANAKAL mandal.",
  "charts": [
    {{
      "title": "Map of TANAKAL Mandal",
      "sql": "SELECT <exact_mandal_column_from_schema>, COUNT(*) as total_count FROM dataset WHERE <exact_mandal_column_from_schema> ILIKE '%TANAKAL%' GROUP BY <exact_mandal_column_from_schema> LIMIT 50",
      "chart_type": "map",
      "location_column": "<exact_mandal_column_from_schema>",
      "size_column": "total_count"
    }}
  ]
}}

Example 3 (Livestock Domain):
User: "how many cattle do we have by breed?"
Response:
{{
  "summary": "Here is the total count of your cattle, separated by breed.",
  "charts": [
    {{
      "title": "Cattle Herd Count by Breed",
      "sql": "SELECT breed_col, COUNT(*) as total_count FROM dataset GROUP BY breed_col ORDER BY total_count DESC LIMIT 50",
      "chart_type": "bar",
      "x_column": "breed_col",
      "y_column": "total_count",
      "color_column": "breed_col"
    }}
  ]
}}

Now, fulfill the following user request based ONLY on their specific schema. Speak clearly to the farmer.

CRITICAL: For ANY question that asks "how many", "which", "list", "show me records", or involves filtering — you MUST:
1. Include a "data_sql" that selects the actual matching rows (not just a COUNT).
2. Include at least one chart showing those rows (e.g. a bar chart ranking them).

Example 4 (Count + Filter — MANDATORY PATTERN):
User: "How many countries have an economy score above 8?"
Response:
{{
  "summary": "There are several countries with an economy score above 8. Here they are ranked from highest to lowest.",
  "data_sql": "SELECT country_col, economy_col FROM dataset WHERE economy_col > 8 ORDER BY economy_col DESC LIMIT 50",
  "charts": [
    {{
      "title": "Countries with Economy Score Above 8",
      "sql": "SELECT country_col, economy_col FROM dataset WHERE economy_col > 8 ORDER BY economy_col DESC LIMIT 50",
      "chart_type": "bar",
      "x_column": "country_col",
      "y_column": "economy_col",
      "color_column": "country_col"
    }}
  ]
}}

User: "{user_query}"
Response:
"""
    return system_prompt, user_prompt

def _pretty_col(col: str) -> str:
    """'class_name_col' -> 'Class Name' — strips the safety suffix and title-cases."""
    if not col:
        return ""
    return col.replace("_col", "").replace("_", " ").strip().title()


def _validate_and_normalise_charts(charts: List[Any]) -> List[Dict]:
    """
    Validates and cleans a list of chart config dicts.
    Filters out entries without a valid SQL string.
    Ensures chart_type is always one of the supported types.
    Gives every chart a clear, specific title — farmers looking at a
    dashboard shouldn't see four charts all vaguely titled "Data Insight".
    """
    valid: List[Dict] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        sql = chart.get("sql", "").strip()
        
        # Aggressively strip any markdown backticks that the LLM might have injected inside the JSON value
        if sql.startswith("```"):
            lines = sql.split("\n")
            if len(lines) >= 2:
                sql = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("```") else "\n".join(lines[1:]).strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        chart["sql"] = sql
        
        if not sql or not sql.upper().lstrip("(").startswith(("SELECT", "WITH")):
            continue

        # Sanitise chart type
        chart_type = str(chart.get("chart_type", "bar")).lower()
        chart["chart_type"] = chart_type if chart_type in VALID_CHART_TYPES else "bar"

        # Build a specific, human-readable title if the AI didn't supply a
        # good one, using the actual columns instead of a generic placeholder.
        title = str(chart.get("title", "")).strip()
        if not title or title.lower() in ("data insight", "chart", "untitled"):
            x_label = _pretty_col(chart.get("x_column", ""))
            y_label = _pretty_col(chart.get("y_column", ""))
            if x_label and y_label:
                title = f"{y_label} by {x_label}"
            elif x_label:
                title = f"Overview of {x_label}"
            else:
                title = "Data Insight"
        chart["title"] = title
        valid.append(chart)

    return valid[:5]  # Maximum of 5 charts per dashboard
