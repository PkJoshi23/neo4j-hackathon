"""
app.py - Enhanced Split/Full-Width Streamlit Dashboard for AI Supply Chain Stock Assistant.
Built for the Relate by Neo4j Hackathon (https://relate.devpost.com/).

Key Capabilities:
1. Full-Width Workspace Mode: Resolves layout squashing by providing dedicated 100% width tabs.
2. The "AI Stack" View: Hierarchical visual strata from Tier 0 (Equipment) to Tier 4 (Hyperscalers).
3. Focus Network (Ego View): 1-to-N hop upstream/downstream exploration.
4. Natural Language to Cypher (Text-to-Graph): Translates English into Cypher 25 on Relate.
5. GraphRAG Analyst Chat: Vector similarity + Relate topology traversal with DeepSeek and SEC auditing.
"""

import os
import sys
import types

# --- Critical Fix for NumPy 2.x vs OpenCV (cv2) compatibility ---
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

import re
import asyncio
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Visualization
from streamlit_agraph import agraph, Node, Edge, Config

# Local Embeddings & LLM
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Relate Graph Engine
from relate_adapter import RelateSupplyChainGraph, TIER_DEFINITIONS, embed_query

load_dotenv()

# Streamlit copies top-level secrets into os.environ, but only once something has actually
# read st.secrets - and it raises when no secrets file exists at all (local dev).
try:
    list(st.secrets.keys())
except Exception:
    pass

# Ingestion is the only writer of the graph and the only reason edgartools is installed, so
# the hosted demo turns it off with INGEST_ENABLED="false" in the secrets tab and runs read-only.
LIVE_INGEST_ENABLED = os.getenv("INGEST_ENABLED", "true").strip().lower() not in ("false", "0", "no")

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Supply Chain Assistant | Relate by Neo4j",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #00C853, #00B4D8, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .relate-badge {
        background-color: #002B36;
        color: #00E676;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        margin-bottom: 0.8rem;
        border: 1px solid #00E676;
    }
    .tier-badge-0 { background-color: #5D4037; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .tier-badge-neg { background-color: #37474F; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .tier-badge-1 { background-color: #2E7D32; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .tier-badge-2 { background-color: #1565C0; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .tier-badge-3 { background-color: #6A1B9A; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .tier-badge-4 { background-color: #E65100; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
    .cypher-box {
        background-color: #1A1A1A;
        color: #69F0AE;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.92rem;
        border-left: 4px solid #00E676;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    /* Chart legend */
    .chart-legend {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        margin: 4px 0 14px 2px;
        font-size: 0.78rem;
        color: #8B949E;
    }
    .chart-legend .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .chart-legend .legend-chip {
        width: 11px;
        height: 11px;
        border-radius: 3px;
        display: inline-block;
    }
    .chart-legend .legend-line {
        width: 26px;
        height: 0;
        display: inline-block;
        border-top: 2px solid #8B949E;
    }
    .chart-legend .legend-line.dashed {
        border-top-style: dashed;
    }
    .chart-legend .legend-line.dotted {
        border-top-style: dotted;
        border-top-width: 3px;
    }
    .chart-legend .legend-sep {
        width: 1px;
        height: 14px;
        background-color: #30363D;
    }
    /* Tab headers styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 6px 6px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_relate_graph():
    """
    Initialize and cache the embedded Relate graph.

    A failed open is returned rather than raised, because Relate is embedded: it takes an exclusive
    lock on one file for the whole life of the process, so "another copy of the app is still
    running" is a normal condition to explain, not a traceback to plaster across the page. The
    cache is cleared on the way out so the next reload retries instead of replaying the failure.
    """
    try:
        # With ingestion switched off nothing in the app writes, so the hosted demo opens the
        # store read-only: no WAL, no lock churn, no boot-time backfill of derived columns.
        return RelateSupplyChainGraph(read_only=not LIVE_INGEST_ENABLED)
    except Exception as exc:
        get_relate_graph.clear()
        return exc


@st.cache_resource
def get_embedder():
    """Loads local SentenceTransformer model (BAAI/bge-small-en-v1.5) for zero-cost embeddings."""
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            return SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception:
            return None
    return None


@st.cache_resource
def get_llm_client():
    """Initializes DeepSeek OpenAI-compatible client."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if HAS_OPENAI and api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None


class SimpleLLMWrapper:
    """Fallback LLM wrapper for DeepSeek calls."""
    def __init__(self, client):
        self.client = client

    def invoke(self, prompt: str) -> str:
        if not self.client:
            return (
                "DeepSeek API key is not configured. Please set `DEEPSEEK_API_KEY` in your `.env` file. "
                "The embedded Relate Knowledge Graph and interactive visualization are fully active."
            )
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"LLM Generation Error: {e}"


TEXT_TO_CYPHER_PROMPT = """
You are an expert Cypher query generator for Relate by Neo4j (Cypher 25).
Translate the user's natural language question into a clean, read-only Cypher query.

Schema, reported by the store right now (labels with their properties, then relationship types):
{schema}

Architectural tiers, read live from the graph:
{tier_roster}

Rules:
1. ONLY read-only Cypher (MATCH, WHERE, RETURN, ORDER BY, LIMIT, WITH, DISTINCT, count, collect).
   The app refuses CREATE, MERGE, SET, DELETE and CALL before a statement reaches the store.
2. Use ONLY the properties listed above. One that is not listed does not exist: Relate answers null
   for it rather than raising, so a WHERE or ORDER BY on it silently does nothing.
3. Return whole nodes (RETURN s, r, t) when a chart helps - the canvas draws :Company nodes.
4. Respond ONLY with the raw Cypher query inside ```cypher ... ``` code block.

User Question:
{user_query}
"""

SYSTEM_ANALYST_PROMPT = """
You are a Senior AI Supply Chain Equity Research Analyst mapping the "AI Money Cycle" 
(Hyperscaler CapEx -> AI Chipmakers -> Foundries, HBM Memory, & Power/Cooling suppliers).

Connected Graph Topology from Relate:
{graph_context}

SEC Filing Excerpts:
{sec_context}

User Question:
{user_query}

Guidelines:
1. Ground your answer in the provided SEC filing excerpts and Relate graph relationships.
2. Clearly distinguish between direct Tier-1 suppliers/customers, capital or partnership links, and 2nd-order bottlenecks described in the filings.
3. If specific metrics are not present, explicitly state that it was not disclosed in the ingested filings.
"""


CYPHER_CORRECTION_PROMPT = """
You are fixing a Cypher query you already wrote. It ran against Relate by Neo4j and failed.

Original question:
{user_query}

Your query:
```cypher
{failed}
```

What went wrong: {reason}

The schema the store reports:
{schema}

Architectural tiers:
{tier_roster}

Rewrite it so it runs and answers the question with fields that exist. Same rules: read-only, only
the listed properties, and no invented field to force a result. If the store genuinely cannot answer
this, write the query that demonstrates the gap (for example MATCH (c:Company) WHERE c.tier = 9
RETURN c LIMIT 1) instead of substituting a different question. Respond ONLY with the corrected
Cypher inside a ```cypher ... ``` block.
"""

CYPHER_MAX_ATTEMPTS = 3
_LLM_UNAVAILABLE = ("DeepSeek API key is not configured", "LLM Generation Error")


def extract_cypher(raw: str) -> Optional[str]:
    """Pull a statement out of the model's reply, or None when it did not write one.

    The old fallback here was a generic SUPPLIES_TO query, so a model that refused or a missing API
    key still drew a chart - of something unrelated to the question. Returning None lets the studio
    say it produced nothing instead.
    """
    if not raw or raw.startswith(_LLM_UNAVAILABLE):
        return None
    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() or None
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("MATCH"):
            return line
    return None


def generate_cypher_query(user_query: str, llm: SimpleLLMWrapper, graph,
                          schema: Optional[str] = None,
                          tier_roster: Optional[str] = None) -> Optional[str]:
    """Uses DeepSeek to translate plain English into a Cypher query the schema supports."""
    prompt = TEXT_TO_CYPHER_PROMPT.format(
        user_query=user_query,
        schema=schema or graph.schema_block(),
        tier_roster=tier_roster or graph.tier_roster(),
    )
    return extract_cypher(llm.invoke(prompt))


def run_cypher_with_correction(user_query: str, llm: SimpleLLMWrapper, graph,
                               max_attempts: int = CYPHER_MAX_ATTEMPTS) -> Dict[str, Any]:
    """
    Generate, run, and if the store objects, generate again with the failure in front of the model.

    Measured on ten questions: DeepSeek produced no syntax errors at all, yet "which companies filed
    a 13G on Nvidia" came back as an INVESTED_IN query with zero rows, and "suppliers of TSMC ordered
    by filing date" returned nine rows sorted by `r.filing_date` - edges carry `as_of`, so Relate
    handed back nulls and the ORDER BY silently did nothing. A rejected query is loud; a wrong one
    that renders a confident table is not. So three outcomes count as failure here: refused, empty,
    and rows whose filters name properties the store does not have.
    """
    profile = graph.schema_profile()
    schema = graph.schema_block(profile)
    roster = graph.tier_roster()
    query = generate_cypher_query(user_query, llm, graph, schema=schema, tier_roster=roster)
    trail: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        if not query:
            return {"trail": trail, "result": {}, "query": None, "attempts": attempt - 1,
                    "error": "No Cypher came back - the model wrote no query block, which is what "
                             "happens when DEEPSEEK_API_KEY is unset or the call fails. The graph "
                             "views do not need it; this studio does."}
        result = graph.execute_cypher(query, profile=profile)
        gaps = result.get("property_gaps") or []
        if result.get("status") == "error":
            outcome = "error"
            reason = "Relate rejected it: " + result["message"].splitlines()[0].strip()
        elif result.get("empty"):
            outcome = "empty"
            reason = "It ran and matched nothing" + (
                " - it reads properties the store does not have: "
                + ", ".join(g["ref"] for g in gaps) if gaps else "") + "."
        elif gaps:
            outcome = "gaps"
            reason = ("It returned rows, but these are not stored: "
                      + "; ".join(f"{g['ref']} on {g['bound_to']}" for g in gaps)
                      + " - so any filter or ordering that used them did nothing.")
        else:
            trail.append({"attempt": attempt, "query": query, "outcome": "ok",
                          "reason": f"{result['count']} row(s) matched.",
                          "rows": result["count"]})
            return {"trail": trail, "result": result, "query": query,
                    "attempts": attempt, "corrected": attempt > 1}

        trail.append({"attempt": attempt, "query": query, "outcome": outcome,
                      "reason": reason, "rows": result.get("count", 0)})
        if attempt >= max_attempts:
            break
        fixed = extract_cypher(llm.invoke(CYPHER_CORRECTION_PROMPT.format(
            user_query=user_query, failed=query, reason=reason,
            schema=schema, tier_roster=roster)))
        if not fixed or fixed.strip().lower() == query.strip().lower():
            break   # no key, or the model repeated itself: a third try is only latency
        query = fixed

    return {"trail": trail, "result": result, "query": query,
            "attempts": len(trail), "corrected": len(trail) > 1}


# -----------------------------------------------------------------------------
# GRAPH RENDERING HELPERS
# -----------------------------------------------------------------------------
# Charts are laid out deterministically: every node gets an authored coordinate and
# physics is only used to fit the canvas. That keeps the picture identical on every
# run instead of drifting into a hairball of overlapping nodes and long edge labels.

# Long component ids get a readable two-line chip label instead of a raw snake_case id.
COMPONENT_LABELS = {
    "EUV_Litho": "EUV\nLithography",
    "Liquid_Cooling": "Direct Liquid\nCooling",
    "CoWoS": "CoWoS\nPackaging",
    "HBM3e": "HBM3e\nMemory",
    "Custom_ASIC": "Custom\nASIC",
    "Optical_Transceiver": "Optical\nTransceiver",
    "Blackwell_B200": "Blackwell\nB200",
}

RELATION_STYLES = {
    "SUPPLIES_TO": {"dashes": False, "legend": "Supplies to", "swatch": "solid"},
    "PRODUCES": {"dashes": True, "legend": "Produces", "swatch": "dashed"},
    "DEPENDS_ON": {"dashes": [3, 5], "legend": "Depends on", "swatch": "dotted"},
    "INVESTED_IN": {"dashes": [8, 4], "legend": "Invests in", "swatch": "dashed"},
    "PARTNERS_WITH": {"dashes": [2, 4], "legend": "Partners with", "swatch": "dotted"},
}

GUTTER_X = 92.0                 # tier label column (left chart axis)
COMPANY_LANE = (186.0, 936.0)   # left/right edge of the company lane
COMPONENT_LANE = (972.0, 1300.0)
COMPANY_STEP = 145.0            # preferred gap between two nodes on a shelf
COMPONENT_STEP = 140.0
SHELF_H = 82.0                  # vertical gap between wrapped rows inside a tier
ROW_GAP = 36.0                  # gap between two tiers
TOP_Y = 70.0
BOTTOM_PAD = 24.0
HEADER_Y = 24.0
MIN_CHART_HEIGHT = 320

FOCUS_COLOR = "#1E88E5"


def _mix(color: str, target: tuple, amount: float) -> str:
    """Blend a hex colour towards an (r, g, b) target by ``amount`` (0..1)."""
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r + (target[0] - r) * amount)),
        int(round(g + (target[1] - g) * amount)),
        int(round(b + (target[2] - b) * amount)),
    )


def _tint(color: str, amount: float) -> str:
    return _mix(color, (255, 255, 255), amount)


def _shade(color: str, amount: float) -> str:
    return _mix(color, (0, 0, 0), amount)


def _component_label(node_id: str) -> str:
    """Readable chip label for a component node id."""
    if node_id in COMPONENT_LABELS:
        return COMPONENT_LABELS[node_id]
    head, _, tail = node_id.partition("_")
    if tail:
        return f"{head}\n{tail.replace('_', ' ')}"
    return node_id


def _row_positions(node_ids, center_x: float, step: float, y: float) -> dict:
    """Evenly spaced, centred horizontal positions in the given order."""
    node_ids = list(node_ids)
    if not node_ids:
        return {}
    start = center_x - (len(node_ids) - 1) * step / 2.0
    return {node_id: (start + i * step, y) for i, node_id in enumerate(node_ids)}


def _spread(node_ids, center_x: float, step: float, y: float) -> dict:
    """Evenly spaced positions on one row, sorted for a stable default order."""
    return _row_positions(sorted(node_ids), center_x, step, y)


def _mean(values):
    return sum(values) / len(values) if values else None


def _shelf_positions(node_ids, lane, step: float, row_top: float, shelf_h: float):
    """
    Wraps one lane into as many shelves as it needs.

    A tier that gains a dozen nodes must not stretch into one giant row (that forces
    vis.js to shrink the whole chart), so nodes flow onto extra shelves inside the
    lane and the canvas grows in height instead.
    """
    node_ids = list(node_ids)
    if not node_ids:
        return {}, 0

    left, right = lane
    capacity = max(1, int((right - left) // step) + 1)
    shelves = (len(node_ids) + capacity - 1) // capacity
    centre = (left + right) / 2.0

    positions = {}
    cursor = 0
    for shelf in range(shelves):
        # Balance the load: 6 nodes over 2 shelves becomes 3 + 3, never 5 + 1.
        remaining = len(node_ids) - cursor
        take = (remaining + (shelves - shelf) - 1) // (shelves - shelf)
        chunk = node_ids[cursor:cursor + take]
        cursor += take

        span = min(step, (right - left) / max(len(chunk) - 1, 1))
        start = centre - (len(chunk) - 1) * span / 2.0
        y = row_top + shelf * shelf_h
        for index, node_id in enumerate(chunk):
            positions[node_id] = (start + index * span, y)
    return positions, shelves


def _chart_config(height: int) -> Config:
    """Shared vis.js options: straight edges, zoom buttons, hover tooltips, fit-to-content."""
    config = Config(width="100%", height=height, directed=True, physics=True, hierarchical=False)
    # Nodes carry explicit positions and are pinned; physics only stabilises + fits the view.
    config.physics = {
        "enabled": True,
        "solver": "repulsion",
        "repulsion": {
            "nodeDistance": 170,
            "centralGravity": 0.0,
            "springLength": 120,
            "springConstant": 0.0,
            "damping": 0.9,
        },
        "minVelocity": 0.5,
        "maxVelocity": 30,
        "timestep": 0.35,
        "stabilization": {"enabled": True, "iterations": 12, "updateInterval": 25, "fit": True},
    }
    config.layout = {"hierarchical": {"enabled": False}, "randomSeed": 7, "improvedLayout": False}
    config.edges = {
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
        "smooth": {"enabled": False},
    }
    config.interaction = {
        "hover": True,
        "tooltipDelay": 90,
        "navigationButtons": True,
        "keyboard": False,
        "multiselect": False,
    }
    return config


def _company_node(meta: dict, x: float, y: float) -> Node:
    color = TIER_DEFINITIONS.get(meta.get("tier", 2), {}).get("color", "#546E7A")
    return Node(
        id=meta["id"],
        label=meta["label"],
        shape="box",
        size=26,
        x=x,
        y=y,
        fixed=True,
        color={
            "background": color,
            "border": _shade(color, 0.25),
            "highlight": {"background": "#FFD54F", "border": "#F57F17"},
            "hover": {"background": _tint(color, 0.15), "border": "#FFD54F"},
        },
        font={"size": 17, "color": "#FFFFFF", "bold": True, "face": "Segoe UI"},
        margin={"top": 9, "bottom": 9, "left": 15, "right": 15},
        borderWidth=2,
        shapeProperties={"borderRadius": 9},
        title=meta.get("title", meta["label"]),
    )


def _component_node(meta: dict, x: float, y: float) -> Node:
    color = TIER_DEFINITIONS.get(meta.get("tier", 2), {}).get("color", "#546E7A")
    return Node(
        id=meta["id"],
        label=_component_label(meta["id"]),
        shape="box",
        size=14,
        x=x,
        y=y,
        fixed=True,
        color={
            "background": _tint(color, 0.84),
            "border": color,
            "highlight": {"background": "#FFE082", "border": "#F57F17"},
            "hover": {"background": _tint(color, 0.7), "border": "#F57F17"},
        },
        font={"size": 13.5, "color": _shade(color, 0.35), "bold": True, "face": "Segoe UI"},
        margin={"top": 7, "bottom": 7, "left": 11, "right": 11},
        borderWidth=1.5,
        shapeProperties={"borderRadius": 6},
        title=meta.get("title", meta["id"]),
    )


def _ego_node(meta: dict, x: float, y: float) -> Node:
    role = meta.get("role", "supplier")
    color = meta.get("color", "#546E7A")
    is_component = role == "component"
    is_focus = role == "focus"
    hop = meta.get("hop") or 0
    size = 30 if is_focus else (14 if is_component else max(20, 26 - hop * 3))
    return Node(
        id=meta["id"],
        label=_component_label(meta["id"]) if is_component else meta["label"],
        shape="box",
        size=size,
        x=x,
        y=y,
        fixed=True,
        color={
            "background": _tint(color, 0.84) if is_component else color,
            "border": FOCUS_COLOR if is_focus else _shade(color, 0.25),
            "highlight": {"background": "#FFD54F", "border": "#F57F17"},
        },
        font={
            "size": 13 if is_component else (17 if is_focus else 15),
            "color": _shade(color, 0.35) if is_component else "#FFFFFF",
            "bold": True,
            "face": "Segoe UI",
        },
        margin={"top": 8, "bottom": 8, "left": 13, "right": 13},
        borderWidth=3 if is_focus else 1.5,
        shapeProperties={"borderRadius": 9},
        title=meta.get("title", meta["label"]),
    )


def _chart_edge(edge: dict) -> Edge:
    style = RELATION_STYLES.get(edge["type"], RELATION_STYLES["SUPPLIES_TO"])
    notes = edge.get("notes") or ""
    form = edge.get("source_form")
    if form:
        provenance = f"{form} · {edge.get('accession_no') or ''} · {edge.get('as_of') or ''}"
    else:
        provenance = "curated baseline (no SEC filing)"
    tooltip = (
        f"{edge['source']} → {edge['target']}\n{edge['type']}\n{provenance}"
        + (f"\n{notes}" if notes else "")
    )
    return Edge(
        source=edge["source"],
        target=edge["target"],
        color={
            "color": _shade(edge["color"], 0.12),
            "opacity": 0.72,
            "highlight": "#FFD54F",
            "hover": "#FFD54F",
        },
        width=1.6,
        selectionWidth=2.2,
        dashes=style["dashes"],
        arrows={"to": {"enabled": True, "scaleFactor": 0.62}},
        title=tooltip,
    )


def _tier_axis_label(tier: int, node_count: int, y: float) -> Node:
    color = TIER_DEFINITIONS.get(tier, {}).get("color", "#546E7A")
    return Node(
        id=f"__tier_axis_{tier}",
        label=f"TIER {tier}\n{TIER_DEFINITIONS.get(tier, {}).get('short', '')}",
        shape="text",
        size=1,
        x=GUTTER_X,
        y=y,
        fixed=True,
        color=color,
        font={"size": 12.5, "color": color, "bold": True, "face": "Segoe UI", "align": "left"},
        title=f"{TIER_DEFINITIONS.get(tier, {}).get('name', f'Tier {tier}')} · {node_count} nodes",
    )


def _column_header(header_id: str, text: str, x: float, y: float) -> Node:
    return Node(
        id=f"__col_{header_id}",
        label=text,
        shape="text",
        size=1,
        x=x,
        y=y,
        fixed=True,
        color="#8B949E",
        font={"size": 11, "color": "#8B949E", "bold": True, "face": "Segoe UI", "align": "center"},
        title=text,
    )


def _render_chart_legend(relation_types=None) -> None:
    """Legend for tier colours plus only the relationship styles present in the chart."""
    tier_html = "".join(
        f'<span class="legend-item"><span class="legend-chip" style="background:{TIER_DEFINITIONS[t]["color"]}">'
        f'</span>Tier {t}</span>'
        for t in sorted(TIER_DEFINITIONS, reverse=True)
    )
    shown = [style for name, style in RELATION_STYLES.items()
             if relation_types is None or name in relation_types]
    relation_html = "".join(
        f'<span class="legend-item"><span class="legend-line {style["swatch"]}"></span>{style["legend"]}</span>'
        for style in shown
    )
    st.markdown(
        f'<div class="chart-legend">{tier_html}<span class="legend-sep"></span>{relation_html}'
        f'<span class="legend-sep"></span>'
        f'<span class="legend-item">Hover a node or edge for detail · drag to pan · scroll to zoom</span></div>',
        unsafe_allow_html=True,
    )


def _render_tiered_chart(nodes_dict, edges_list, min_height, with_tier_axis=True):
    """
    Draws a tiered supply-chain chart.

    Companies sit in the left lane, components in the right lane, and each lane wraps
    onto extra shelves inside its tier when it grows. The canvas height follows the
    content, so a tier that gains nodes makes the chart taller instead of forcing
    vis.js to shrink everything until the labels are unreadable. Edge text labels are
    replaced by hover tooltips and relationship line styles.
    """
    if not nodes_dict:
        return

    present_tiers = sorted({meta.get("tier", 2) for meta in nodes_dict.values()}, reverse=True)
    lanes = {tier: {"companies": [], "components": []} for tier in present_tiers}
    for nid, meta in nodes_dict.items():
        tier = meta.get("tier", 2)
        lanes.setdefault(tier, {"companies": [], "components": []})
        lane = "components" if meta.get("type") == "Component" else "companies"
        lanes[tier][lane].append(nid)

    # Components were retired; when none exist the company lane takes the whole width.
    has_components = any(meta.get("type") == "Component" for meta in nodes_dict.values())
    company_lane = COMPANY_LANE if has_components else (COMPANY_LANE[0], COMPONENT_LANE[1])
    company_centre = (company_lane[0] + company_lane[1]) / 2.0

    # Crossings hurt readability, so each lane is ordered with a barycentre sweep:
    # every node slides towards the average x of the nodes it is connected to.
    neighbours = {}
    for edge in edges_list:
        neighbours.setdefault(edge["source"], set()).add(edge["target"])
        neighbours.setdefault(edge["target"], set()).add(edge["source"])

    lane_geometry = {
        "companies": (company_centre, COMPANY_STEP),
        "components": ((COMPONENT_LANE[0] + COMPONENT_LANE[1]) / 2.0, COMPONENT_STEP),
    }
    x_of = {}
    for tier in present_tiers:
        for lane, (centre, step) in lane_geometry.items():
            for nid, (x, _y) in _spread(lanes[tier][lane], centre, step, 0.0).items():
                x_of[nid] = x

    def _barycenter(node_id: str) -> float:
        placed = [x_of[n] for n in neighbours.get(node_id, ()) if n in x_of]
        if placed:
            return _mean(placed)
        # Nodes with no relationships keep out of the way at the end of the row.
        return float("inf")

    for pass_index in range(3):
        tier_sequence = present_tiers if pass_index % 2 == 0 else list(reversed(present_tiers))
        for tier in tier_sequence:
            for lane, (centre, step) in lane_geometry.items():
                members = lanes[tier][lane]
                if len(members) < 2:
                    continue
                members.sort(key=lambda nid: (_barycenter(nid), x_of.get(nid, 0.0)))
                for nid, (x, _y) in _row_positions(members, centre, step, 0.0).items():
                    x_of[nid] = x

    # Shelf layout: the ordered lane is chunked into rows inside its lane, and the
    # canvas grows with the content instead of scaling the whole chart down.
    positions = {}
    row_centre = {}
    y_cursor = TOP_Y
    for tier in present_tiers:
        company_positions, company_shelves = _shelf_positions(
            lanes[tier]["companies"], company_lane, COMPANY_STEP, y_cursor, SHELF_H
        )
        component_positions, component_shelves = _shelf_positions(
            lanes[tier]["components"], COMPONENT_LANE, COMPONENT_STEP, y_cursor, SHELF_H
        )
        positions.update(company_positions)
        positions.update(component_positions)
        shelves = max(company_shelves, component_shelves, 1)
        row_centre[tier] = y_cursor + (shelves - 1) * SHELF_H / 2.0
        y_cursor += shelves * SHELF_H + ROW_GAP

    height = max(MIN_CHART_HEIGHT, int(min_height), int(y_cursor - ROW_GAP + BOTTOM_PAD))

    agraph_nodes = []
    if with_tier_axis:
        for tier in present_tiers:
            count = sum(1 for meta in nodes_dict.values() if meta.get("tier") == tier)
            agraph_nodes.append(_tier_axis_label(tier, count, row_centre[tier]))

    agraph_nodes.append(_column_header("companies", "COMPANIES", lane_geometry["companies"][0], HEADER_Y))
    if has_components:
        agraph_nodes.append(
            _column_header("components", "COMPONENTS & CAPABILITIES", lane_geometry["components"][0], HEADER_Y)
        )

    for nid, meta in nodes_dict.items():
        x, y = positions[nid]
        if meta.get("type") == "Component":
            agraph_nodes.append(_component_node(meta, x, y))
        else:
            agraph_nodes.append(_company_node(meta, x, y))

    agraph_edges = [_chart_edge(edge) for edge in edges_list]
    agraph(nodes=agraph_nodes, edges=agraph_edges, config=_chart_config(height))
    _render_chart_legend({edge["type"] for edge in edges_list})


def render_query_log(relate_graph, limit: int = 6) -> None:
    """
    Shows the Cypher statements Relate actually executed for this view.

    Everything on screen is read from the embedded graph database, so this panel is the
    receipt: query text, row count and latency for the most recent statements.
    """
    queries = relate_graph.recent_queries()[:limit]
    if not queries:
        return

    with st.expander(f"🔎 Cypher executed against Relate (last {len(queries)})", expanded=False):
        for index, query in enumerate(queries, start=1):
            if query["error"]:
                status = f"⚠️ error · {query['error'][:90]}"
            else:
                status = f"{query['rows']} row(s) · {query['ms']} ms"
            st.caption(f"{index}. {status}")
            st.code(query["cypher"], language="cypher")


def render_tiered_stack(relate_graph, selected_tiers, height=680):
    """Renders the AI Stack as a tiered chart (Tier 4 top → Tier 0 bottom)."""
    nodes_dict, edges_list, tier_counts = relate_graph.get_tiered_stack(selected_tiers)
    summaries = relate_graph.get_tier_summaries()

    # Top metrics bar (one column per tier, highest tier first)
    tier_order = sorted(TIER_DEFINITIONS, reverse=True)
    m_cols = st.columns(len(tier_order))
    for col, tier in zip(m_cols, tier_order):
        col.metric(label=TIER_DEFINITIONS[tier]["name"].split(":")[0], value=tier_counts.get(tier, 0))

    if not nodes_dict:
        st.warning("No tiers selected. Please check at least one tier in the controls above or sidebar.")
        return

    if summaries:
        with st.expander("Per-tier rollup (read from the tier nodes)", expanded=False):
            rows = [{
                "Tier": item["tier"],
                "Stratum": (item.get("short") or item.get("name") or "").replace("\n", " "),
                "Members": item.get("members", 0),
                "Companies": ", ".join(item.get("tickers") or []),
                "Feeds": ", ".join(f"T{f['tier']} ({f['edges']})" for f in item["feeds"]) or "-",
                "Fed by": ", ".join(f"T{f['tier']} ({f['edges']})" for f in item["fed_by"]) or "-",
                "Out edges": item["out_edges"],
                "Cited": f"{item['evidence_pct']}%" if item["evidence_pct"] is not None else "n/a",
            } for item in summaries
                if selected_tiers is None or item["tier"] in selected_tiers]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Every number is aggregated live from the company edges rather than stored on the "
                "tier, so a stratum cannot report a count the graph no longer backs. 'Cited' is the "
                "share of that tier's outgoing relationships carrying an SEC accession number; the "
                "rest were curated by hand."
            )

    _render_tiered_chart(nodes_dict, edges_list, height, with_tier_axis=True)


def render_ego_network(relate_graph, focus_ticker, hops, height=680):
    """
    Renders the ego network as a left-to-right supply-chain rail:
    upstream suppliers on the left, the focus company in the middle, downstream
    customers on the right.
    """
    nodes_dict, edges_list = relate_graph.get_topology(focus_ticker, hops)

    if not nodes_dict:
        st.info(f"No graph data found for **{focus_ticker}**. Ingest filings using the sidebar widget.")
        return

    column_x = {
        ("supplier", 2): 165.0,
        ("supplier", 1): 430.0,
        ("focus", 0): 705.0,
        ("customer", 1): 980.0,
        ("customer", 2): 1255.0,
    }
    column_headers = {
        ("supplier", 2): "SUPPLIERS\n2 HOPS",
        ("supplier", 1): "SUPPLIERS\n1 HOP",
        ("focus", 0): "FOCUS",
        ("customer", 1): "CUSTOMERS\n1 HOP",
        ("customer", 2): "CUSTOMERS\n2 HOPS",
    }
    columns = {key: [] for key in column_x}
    components = []
    partners = []
    for nid, meta in nodes_dict.items():
        role = meta.get("role", "supplier")
        hop = meta.get("hop") or 0
        if role == "component":
            components.append(nid)
            continue
        if role == "partner":
            partners.append(nid)
            continue
        key = ("focus", 0) if role == "focus" else (role, min(hop, 2))
        columns.setdefault(key, []).append(nid)

    # The tallest column sets the canvas height, so a hub with a dozen customers gets
    # a taller chart instead of shrinking every label to fit a fixed box.
    tallest = max((len(ids) for ids in columns.values()), default=0)
    top_y = 118.0
    bottom_y = top_y + max(0, tallest - 1) * 78.0
    partner_row_y = bottom_y + 108.0
    component_row_y = partner_row_y + (96.0 if partners else 0.0)
    lowest_row = component_row_y if components else (partner_row_y if partners else bottom_y)
    height = max(MIN_CHART_HEIGHT, int(height), int(lowest_row + 70.0))

    agraph_nodes = []
    y_of = {}

    def _column_ys(count: int):
        if count <= 1:
            return [(top_y + bottom_y) / 2.0]
        step = (bottom_y - top_y) / (count - 1)
        return [top_y + i * step for i in range(count)]

    # Initial alphabetical order, then a vertical barycentre sweep so each node is
    # pulled towards the average height of the companies it trades with.
    neighbours = {}
    for edge in edges_list:
        neighbours.setdefault(edge["source"], set()).add(edge["target"])
        neighbours.setdefault(edge["target"], set()).add(edge["source"])

    for key, ids in columns.items():
        for nid, y in zip(sorted(ids), _column_ys(len(ids))):
            y_of[nid] = y

    ordered_columns = {key: sorted(ids) for key, ids in columns.items() if ids}
    for pass_index in range(3):
        sequence = list(ordered_columns) if pass_index % 2 == 0 else list(reversed(ordered_columns))
        for key in sequence:
            members = ordered_columns[key]
            if len(members) < 2:
                continue
            members.sort(
                key=lambda nid: (
                    _mean([y_of[n] for n in neighbours.get(nid, ()) if n in y_of]) or y_of.get(nid, 0.0),
                    y_of.get(nid, 0.0),
                )
            )
            for nid, y in zip(members, _column_ys(len(members))):
                y_of[nid] = y

    for key, ids in ordered_columns.items():
        agraph_nodes.append(_column_header(f"{key[0]}_{key[1]}", column_headers[key], column_x[key], HEADER_Y + 14))
        for nid in ids:
            agraph_nodes.append(_ego_node(nodes_dict[nid], column_x[key], y_of[nid]))

    if partners:
        agraph_nodes.append(_column_header("partners", "PARTNERS & INVESTORS", 740.0, partner_row_y - 62))
        for nid, (x, y) in _spread(partners, 740.0, 195.0, partner_row_y).items():
            agraph_nodes.append(_ego_node(nodes_dict[nid], x, y))

    if components:
        agraph_nodes.append(_column_header("components", "COMPONENTS & CAPABILITIES", 740.0, component_row_y - 62))
        for nid, (x, y) in _spread(components, 740.0, 195.0, component_row_y).items():
            agraph_nodes.append(_ego_node(nodes_dict[nid], x, y))

    agraph_edges = [_chart_edge(edge) for edge in edges_list]
    agraph(nodes=agraph_nodes, edges=agraph_edges, config=_chart_config(height))
    _render_chart_legend({edge["type"] for edge in edges_list})



def _render_tool_trace(message) -> None:
    """Renders the agent's plan -> tool -> observation loop."""
    trace = message.get("trace") or []
    if not trace:
        return
    tokens = message.get("tokens") or 0
    with st.expander(f"🔧 Tool trace · {len(trace)} call(s) · ~{tokens:,} tokens", expanded=False):
        for step in trace:
            icon = "⚠️" if step.get("error") else "✅"
            tokens = f" · {step['tokens']:,} tok" if step.get("tokens") else ""
            st.markdown(
                f"{icon} **{step['step']}. `{step['tool']}`** — {step['summary']} · {step['ms']} ms{tokens}"
            )
            if step["args"].get("cypher"):
                st.code(step["args"]["cypher"], language="cypher")
            elif step["args"]:
                st.caption(", ".join(f"{k}={v}" for k, v in step["args"].items()))

    citations = message.get("citations") or []
    if citations:
        with st.expander(f"📎 Filing evidence ({len(citations)})", expanded=False):
            for citation in citations:
                st.markdown(
                    f"**{citation.get('ticker')}** · `{citation.get('accession_no')}`\n\n"
                    f"> {(citation.get('excerpt') or '')[:220]}"
                )


def render_chat_interface(relate_graph, embedder, llm):
    """Chat tab: tool-using agent (default) or the fast one-shot GraphRAG analyst."""
    mode = st.radio(
        "Chat mode",
        ["🤖 Agent (plans + calls tools)", "⚡ GraphRAG one-shot (fast)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode.startswith("🤖"):
        _render_agent_chat(relate_graph, embedder, llm)
    else:
        _render_graphrag_chat(relate_graph, embedder, llm)


def _render_agent_chat(relate_graph, embedder, llm):
    """
    Multi-step analyst agent: it plans, queries Relate with Cypher, reads SEC excerpts and
    searches filings, then answers with citations. The whole loop is shown to the user.
    """
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = [
            {
                "role": "assistant",
                "content": (
                    "I am a **tool-using supply-chain analyst**. I can run Cypher against the "
                    "embedded Relate graph, expand any company's network, read the ingested SEC "
                    "filings, and ingest a new ticker if something is missing. Ask me a question "
                    "and watch the tool trace."
                ),
                "trace": [],
                "citations": [],
                "tokens": 0,
            }
        ]

    st.markdown("**Suggested agent missions:**")
    mission_cols = st.columns(3)
    preset_mission = None
    if mission_cols[0].button("🤝 NVIDIA partners + evidence", use_container_width=True):
        preset_mission = "Which companies does NVIDIA partner with, and what filing evidence supports each relationship?"
    if mission_cols[1].button("🔎 Is ORCL connected to NVDA?", use_container_width=True):
        preset_mission = "Is Oracle connected to NVIDIA in this graph? Cite the exact filing evidence."
    if mission_cols[2].button("🕸️ TSMC → hyperscalers", use_container_width=True):
        preset_mission = "Trace the supply path from TSMC up to the hyperscalers and name each hop."

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_tool_trace(message)

    active_prompt = preset_mission or st.chat_input(
        "Ask the agent to trace a supply chain or find filing evidence..."
    )

    if active_prompt:
        st.session_state.agent_messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent planning · querying Relate · reading SEC filings..."):
                from agent import SupplyChainAgent

                agent = SupplyChainAgent(getattr(llm, "client", None), relate_graph, embedder, llm,
                                         allow_ingest=LIVE_INGEST_ENABLED)
                result = agent.run(active_prompt)
                st.markdown(result["answer"])
                _render_tool_trace(result)

                st.session_state.agent_messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "trace": result["trace"],
                    "citations": result["citations"],
                    "tokens": result["tokens"],
                })


def _render_graphrag_chat(relate_graph, embedder, llm):
    """Renders the GraphRAG conversational equity analyst interface."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Supply Chain Research Analyst powered by **Relate by Neo4j**. Ask me about upstream bottlenecks, customer concentrations, or second-order supply chain risks.",
                "audit": None,
            }
        ]

    # Quick Suggestion Chips
    st.markdown("**Suggested Institutional Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    preset_query = None
    if q_col1.button("🔴 What are Nvidia's CoWoS packaging risks?", use_container_width=True):
        preset_query = "What are Nvidia's single-source risks regarding TSMC CoWoS packaging?"
    if q_col2.button("🔵 How does Hyperscaler CapEx flow to foundries?", use_container_width=True):
        preset_query = "Trace the capital flow from Microsoft and Amazon CapEx down to TSMC and ASML."
    if q_col3.button("🟡 What cooling bottlenecks affect Supermicro?", use_container_width=True):
        preset_query = "Explain the liquid cooling supply chain dependencies between Vertiv and Supermicro."

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audit"):
                with st.expander("🔍 SEC Citations & Relate Graph Context"):
                    st.markdown(msg["audit"])

    # Chat input or preset trigger
    user_chat_prompt = st.chat_input("Ask about supply chain bottlenecks, CapEx flow, or risks...")
    active_prompt = preset_query or user_chat_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt, "audit": None})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Traversing Relate graph relationships and SEC excerpts..."):
                q_embedding = []
                if embedder is not None:
                    try:
                        q_embedding = embed_query(embedder, active_prompt)
                    except Exception:
                        q_embedding = []

                rag_data = relate_graph.search_graphrag_context(active_prompt, q_embedding, top_k=3)
                retrieved_chunks = rag_data.get("chunks", [])
                graph_contexts = rag_data.get("graph_contexts", [])

                sec_lines = []
                audit_lines = []
                for idx, ch in enumerate(retrieved_chunks):
                    text = ch.get("text", "")
                    label = " / ".join(str(part) for part in (
                        ch.get("ticker"), ch.get("form"), ch.get("as_of")) if part) or "unknown issuer"
                    cosine = ch.get("similarity")
                    confidence = f", cosine {cosine:.3f}" if isinstance(cosine, (int, float)) else ""
                    sec_lines.append(
                        f"Excerpt {idx+1} ({label}, accession {ch.get('accession_no')}{confidence}):\n{text}")
                    audit_lines.append(f"**SEC Filing Excerpt {idx+1} ({label}{confidence}):**\n> {text}\n")

                unknown_names = rag_data.get("unknown_entities") or []
                if sec_lines:
                    sec_context_str = "\n\n".join(sec_lines)
                elif unknown_names:
                    # The honest answer to a question about a company nobody has ingested.
                    sec_context_str = (
                        "No ingested filing mentions " + ", ".join(unknown_names) + ". It has not "
                        "been ingested, so say that plainly instead of describing a different "
                        "company's disclosures as if they were its own."
                    )
                else:
                    sec_context_str = (
                        "No filing excerpt cleared the relevance threshold for this question. State "
                        "that the graph holds no evidence for it rather than answering anyway."
                    )

                graph_lines = []
                for gc in graph_contexts:
                    g_str = (
                        f"- Company: **{gc.get('ticker')}**\n"
                        f"  - Upstream Suppliers: {', '.join(gc.get('suppliers', [])) or 'None recorded'}\n"
                        f"  - Downstream Customers: {', '.join(gc.get('customers', [])) or 'None recorded'}\n"
                        f"  - Critical Dependencies: {', '.join(gc.get('dependencies', [])) or 'None recorded'}"
                    )
                    graph_lines.append(g_str)
                    audit_lines.append(f"**Relate Graph Path:**\n{g_str}\n")

                graph_context_str = "\n".join(graph_lines) if graph_lines else "Graph traversal active."

                full_prompt = SYSTEM_ANALYST_PROMPT.format(
                    graph_context=graph_context_str,
                    sec_context=sec_context_str,
                    user_query=active_prompt,
                )

                ai_text = llm.invoke(full_prompt)
                audit_report = "\n".join(audit_lines) if audit_lines else "Direct Relate graph query executed."

                if unknown_names:
                    st.warning(
                        "Not in the graph: " + ", ".join(f"`{name}`" for name in unknown_names)
                        + " - no filing has been ingested for them, so nothing below is sourced "
                        "from their filings. Add the ticker in the ingestion panel to change that."
                    )
                elif not retrieved_chunks:
                    st.info("No filing excerpt cleared the relevance threshold for this question.")

                st.markdown(ai_text)
                with st.expander("🔍 SEC Citations & Relate Graph Context"):
                    st.markdown(audit_report)

                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_text, "audit": audit_report}
                )


def render_attempt_trail(trail: List[Dict[str, Any]]) -> None:
    """
    Show every generated query and why the earlier ones were not good enough.

    Hiding the retries would make the loop look like luck. The rejected attempts are the evidence
    that the studio checks the store instead of trusting the model.
    """
    if not trail:
        return
    if len(trail) > 1:
        st.caption(f"The studio ran {len(trail)} attempts. Each one below carries the failure the "
                   "model was given before it wrote the next query.")
    for step in trail:
        badge = {"ok": "accepted", "error": "rejected by Relate",
                 "empty": "matched nothing", "gaps": "named a property that is not stored"}[step["outcome"]]
        if step["outcome"] == "ok":
            with st.expander(f"Attempt {step['attempt']} - {badge} ({step['rows']} rows)", expanded=False):
                st.code(step["query"], language="cypher")
            continue
        st.error(f"**Attempt {step['attempt']} - {badge}.** {step['reason']}")
        st.code(step["query"], language="cypher")


def render_text_to_cypher(relate_graph, llm, height=480):
    """Renders the Natural Language to Cypher studio."""
    st.subheader("🔮 Natural Language to Cypher Studio (Text-to-Graph)")
    st.caption("Ask questions in plain English; DeepSeek translates into Cypher 25 executed against Relate.")

    preset_query = st.selectbox(
        "Quick Query Presets:",
        [
            "Custom Query...",
            "Who supplies directly to Microsoft and Amazon?",
            "Which companies has NVIDIA invested in?",
            "Show all suppliers and partners of NVIDIA",
            "Which equipment and cooling suppliers feed into Tier 1 and 3?",
            "Which companies filed a 13G on Nvidia, and when?",
            "Show the suppliers of TSMC ordered by filing date",
        ],
    )

    default_text = "" if preset_query == "Custom Query..." else preset_query
    user_nl_query = st.text_input("Enter natural language question:", value=default_text)

    if st.button("Generate & Run Cypher", type="primary", use_container_width=True):
        if user_nl_query:
            with st.spinner("Translating, executing, and self-correcting against Relate..."):
                outcome = run_cypher_with_correction(user_nl_query, llm, relate_graph)
                render_attempt_trail(outcome["trail"])
                cypher_res = outcome.get("result") or {}

                if outcome.get("error"):
                    st.error(outcome["error"])
                elif cypher_res.get("status") == "error":
                    st.error("Relate still refuses it - the reasons above are what it said each time.")
                elif not cypher_res.get("count"):
                    st.warning(
                        f"No records after {len(outcome['trail'])} attempt(s). That is a finding, not a "
                        "glitch: the graph holds nothing these queries can name. Add the company or the "
                        "filing in the ingestion panel, or ask about what is stored."
                    )
                else:
                    st.success(
                        f"Matched {cypher_res['count']} graph records from Relate"
                        + ("." if len(outcome["trail"]) == 1
                           else f" - corrected on attempt {len(outcome['trail'])}.")
                    )
                    if cypher_res.get("property_gaps"):
                        st.warning(
                            "Read this table with care: "
                            + ", ".join(f"`{gap['ref']}`" for gap in cypher_res["property_gaps"])
                            + " is not a property the store has, so any filter or ordering that named it "
                            "did nothing. The rows themselves are real."
                        )

                    # Subgraph visualization: same tiered chart as the AI Stack view.
                    if cypher_res["nodes"]:
                        _render_tiered_chart(
                            cypher_res["nodes"],
                            cypher_res["edges"],
                            height,
                            with_tier_axis=True,
                        )
                    else:
                        st.info(
                            "Rows matched, but none of them are companies - the canvas draws the "
                            "company subgraph only, so a query over `:Tier` or `:Filing` shows up in "
                            "the table below. Traverse back to companies (for example "
                            "`MATCH (t:Tier {tier: 1})<-[:IN_TIER]-(c:Company) RETURN c`) to draw it."
                        )

                    # Tabular display
                    with st.expander("📊 View Matched Query Records", expanded=True):
                        df = pd.DataFrame(cypher_res["records"])
                        st.dataframe(df, use_container_width=True)

                    render_query_log(relate_graph, limit=4)
        else:
            st.warning("Please enter a question or select a preset.")


# -----------------------------------------------------------------------------
# MAIN APPLICATION ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
    relate_graph = get_relate_graph()
    if isinstance(relate_graph, Exception):
        locked = "lock" in str(relate_graph).lower()
        st.error("The Relate store is already open in another process." if locked
                 else "The Relate store could not be opened.")
        if locked:
            st.markdown(
                "Relate is an **embedded** database: one file, one writer, and the lock belongs to "
                "whoever opened it first. So this page is not broken - another copy of the app, or a "
                "`python ingest.py` / `backfill_chunks.py` run, still holds "
                "`supply_chain.rl`.\n\n"
                "1. `Ctrl+C` the terminal running the other instance (closing the browser tab is not "
                "enough - the server process is what holds the lock).\n"
                "2. If no such terminal is open, a process from a crashed run is stale: end it in "
                "Task Manager.\n"
                "3. Reload this page. It retries on every reload."
            )
        st.caption(f"Relate reported: {relate_graph}")
        st.stop()

    embedder = get_embedder()
    openai_client = get_llm_client()
    llm = SimpleLLMWrapper(openai_client)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown('<div class="relate-badge">⚡ Engine: Relate by Neo4j (Embedded)</div>', unsafe_allow_html=True)
        st.caption(
            f"Graph in Relate: **{relate_graph.count_nodes()}** companies · "
            f"**{relate_graph.count_edges()}** relationships · "
            f"**{relate_graph.count_filings()}** filings"
        )

        st.header("🎛️ Graph Parameters")

        graph_style = st.selectbox(
            "Graph Mode:",
            [
                "🏢 The 'AI Stack' (Hierarchical Strata)",
                "🔍 Focus Network (Ego View)",
            ],
        )

        if graph_style == "🔍 Focus Network (Ego View)":
            focus_ticker = st.text_input("Focus Ticker", value="NVDA").upper().strip()
            hops = st.slider("Supply Chain Hops", min_value=1, max_value=3, value=1)
            selected_tiers = [-1, 0, 1, 2, 3, 4]
        else:
            focus_ticker = "NVDA"
            hops = 1
            st.caption("Active Tiers Filter:")
            selected_tiers = []
            for t_idx in sorted(TIER_DEFINITIONS, reverse=True):
                t_meta = TIER_DEFINITIONS[t_idx]
                if st.checkbox(t_meta["name"], value=True, key=f"t_check_{t_idx}"):
                    selected_tiers.append(t_idx)

        st.markdown("---")
        st.subheader("📥 Live SEC Ingestion")
        st.caption("Scans 10-K/20-F, 8-K, 13D/13G and DEF 14A with a zero-token sieve.")
        ingest_ticker_input = st.text_input(
            "Enter Ticker to Ingest", placeholder="e.g. TSM, ASML, MU",
            disabled=not LIVE_INGEST_ENABLED,
        ).upper().strip()

        if st.button("Ingest Filing", use_container_width=True, disabled=not LIVE_INGEST_ENABLED):
            if ingest_ticker_input:
                with st.spinner(f"Ingesting latest SEC filing for {ingest_ticker_input} into Relate..."):
                    try:
                        from ingest import efficient_ingest_ticker
                        res = asyncio.run(efficient_ingest_ticker(ingest_ticker_input, relate_graph, llm, embedder))
                        if res["status"] == "success":
                            st.success(res["message"])
                            st.rerun()
                        elif res["status"] == "cached":
                            st.info(res["message"])
                        else:
                            st.warning(res["message"])
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
            else:
                st.warning("Please enter a valid ticker.")

        if not LIVE_INGEST_ENABLED:
            st.caption(
                "This deployment opens Relate read-only, so new filings cannot be added here. "
                "The 187 filings and 879 excerpts on screen were ingested by this same code path."
            )

        st.markdown("---")
        st.markdown(
            """
            **Tier Legend:**
            - <span class="tier-badge-4">Tier 4</span> Hyperscalers & AI Cloud
            - <span class="tier-badge-3">Tier 3</span> Server Integrators & Systems
            - <span class="tier-badge-2">Tier 2</span> AI Accelerators & Silicon
            - <span class="tier-badge-1">Tier 1</span> Foundries, Packaging & HBM
            - <span class="tier-badge-0">Tier 0</span> Equipment, Power & Cooling
            - <span class="tier-badge-neg">Tier -1</span> Raw Materials & Specialty Inputs
            """,
            unsafe_allow_html=True,
        )

    # --- Header ---
    st.markdown('<div class="main-title">⚡ AI Supply Chain Stock Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Mapping the AI Money Cycle: Hyperscalers ➔ Accelerators ➔ Foundries & Infrastructure | Built with <b>Relate by Neo4j</b></div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # FULL-WIDTH WORKSPACE (TABS)
    # =========================================================================
    tab_graph, tab_chat, tab_cypher = st.tabs([
        "🌐 Interactive Graph & AI Stack",
        "💬 Supply Chain Agent & Analyst",
        "🔮 Natural Language to Cypher Studio",
    ])

    with tab_graph:
        if graph_style == "🏢 The 'AI Stack' (Hierarchical Strata)":
            st.subheader("🏢 The AI Stack: Hierarchical Supply Chain Strata (Full Width)")
            render_tiered_stack(relate_graph, selected_tiers, height=620)
        else:
            st.subheader(f"🌐 Relate Topology: {focus_ticker} (Hops: {hops}) - Full Width")
            render_ego_network(relate_graph, focus_ticker, hops, height=620)
        render_query_log(relate_graph)

        st.markdown("#### 🧱 Layer Concentration")
        layer_rows = relate_graph.get_layer_map()
        if layer_rows:
            st.dataframe(
                [
                    {
                        "Layer": row["layer"],
                        "Companies": row["members"],
                        "Suppliers": row["suppliers_with_edges"],
                        "Buyer links": row["supply_edges"],
                        "Largest supplier": row["top_supplier"],
                        "Its buyers": row["top_supplier_buyers"],
                        "Load share": f"{int(row['load_share'] * 100)}%",
                        "Cited by a filing": f"{row['evidence_pct']:.0f}%",
                    }
                    for row in layer_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Load share is the fraction of that layer's buyer relationships carried by "
                "its single largest supplier: 80%+ means one company, not a competitive "
                "layer. The last column is honesty - how much of the layer is quoted from a "
                "filing rather than curated by hand."
            )

    with tab_chat:
        st.subheader("💬 Supply Chain Agent & GraphRAG Analyst (Full Width)")
        st.caption(
            "The agent plans, runs Cypher against Relate and reads SEC filings before answering — "
            "with its tool trace and cited accession numbers on screen."
        )
        render_chat_interface(relate_graph, embedder, llm)

    with tab_cypher:
        render_text_to_cypher(relate_graph, llm, height=480)


if __name__ == "__main__":
    main()
