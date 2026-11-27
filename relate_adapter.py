"""
relate_adapter.py - Relate-backed graph store for the AI Supply Chain Stock Assistant.

Relate (embedded graph database, `relate-db`) is the single source of truth: every read
and every write is a Cypher statement. `supply_chain.json` is only an export/debug copy.

Threading: `relate.threadsafety == 1`, so a connection may only be used by the thread
that opened it. Streamlit runs each rerun on a fresh script thread, so the connection is
owned by a single-worker executor and every query is dispatched there. The same worker
records each query, which powers the "Queries executed" panel in the UI.
"""

import atexit
import hashlib
import json
import logging
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from entities import KNOWN_ALIASES

logger = logging.getLogger(__name__)

try:
    import relate
    HAS_RELATE = True
    logger.info("Native 'relate-db' package loaded successfully.")
except ImportError:
    HAS_RELATE = False
    logger.warning("Package 'relate-db' not found. Install it from https://relatedb.ai")

DEFAULT_RL_PATH = os.path.join(os.path.dirname(__file__), "supply_chain.rl")
DEFAULT_CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "supply_chain_chunks.json")

# Filing excerpts are embedded with BAAI/bge-small-en-v1.5. The Relate vector index is
# built for exactly this width, so a chunk with any other dimensionality cannot be indexed
# and is skipped with a warning rather than silently corrupting search.
CHUNK_EMBEDDING_DIM = 384

# bge-small-en-v1.5 was trained asymmetrically: a passage goes in bare, a question goes in behind
# this instruction. The corpus is already embedded bare, so a query without the prefix is compared
# against passages in a vector space the checkpoint never intended for that pairing.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embed_query(embedder, text: str) -> List[float]:
    """Embed a retrieval query the way the checkpoint was trained to be asked."""
    return list(embedder.encode(QUERY_INSTRUCTION + str(text)).tolist())


# Reciprocal Rank Fusion constant: dampens the gap between rank 1 and rank 2 so a chunk
# that both indexes like has more pull than one that tops a single list.
RRF_K = 60

# Cosine gate on what counts as evidence. Measured against the live corpus: off-topic questions
# ("tell me about the coffee machine in the lobby", "how do I reset my password", "zzz banana
# flute") top out at 0.785, while every real supply-chain question bottomed at 0.819. RRF cannot
# express "nothing matched" - the best of five unrelated chunks still scores 1/(60+1) - so without
# this gate the model is handed confident-looking excerpts from the wrong company and answers them.
MIN_VECTOR_SIM = 0.80

# Words a question opens with that are never a company name. Capitalised-sentence-start is the
# only thing marking them, and an advisory "X is not in the graph" naming the word "Which" is
# worse than saying nothing.
NAME_STOPWORDS = frozenset("""
what which who whose whom why how when where does do did is are was were can could should would
will tell show describe explain list give compare any some all the a an and but for from with
about my our its their it they this that these those filing filings company companies stock peer
sector supply supplies customer customers risk risks quarter year revenue margins main key
top-level tier segment segments upstream downstream direct largest biggest item exhibit section
""".split())

# Architectural Tier Definitions for the AI Stack View.
# Tier -1 sits *below* equipment: the materials and specialty inputs the whole stack consumes.
# `short` carries the line break the tier axis needs, so the gutter label and the graph node read
# the same authored string instead of a second copy of it in the UI.
TIER_DEFINITIONS = {
    -1: {"name": "Tier -1: Raw Materials & Specialty Inputs", "short": "Raw Materials\n& Specialty Inputs",
         "color": "#37474F", "edge_color": "#607D8B"},
    0: {"name": "Tier 0: Equipment, Power & Facilities", "short": "Equipment,\nPower & Cooling",
         "color": "#5D4037", "edge_color": "#8D6E63"},
    1: {"name": "Tier 1: Foundries, Packaging & Memory", "short": "Foundries, Packaging\n& Memory",
         "color": "#2E7D32", "edge_color": "#43A047"},
    2: {"name": "Tier 2: AI Accelerators & Silicon Design", "short": "AI Accelerators\n& Silicon",
         "color": "#1565C0", "edge_color": "#1E88E5"},
    3: {"name": "Tier 3: AI Server Integrators & Systems", "short": "Server Integrators",
         "color": "#6A1B9A", "edge_color": "#8E24AA"},
    4: {"name": "Tier 4: Hyperscalers & AI Cloud", "short": "Hyperscalers\n& AI Cloud",
         "color": "#E65100", "edge_color": "#FB8C00"},
}


def tier_node_id(tier: int) -> str:
    return f"TIER_{tier}"

DEFAULT_TIER = 2

# Capital / partnership links get their own colours so they never read as a supply flow.
RELATION_EDGE_COLORS = {
    "INVESTED_IN": "#8E24AA",
    "PARTNERS_WITH": "#00897B",
}

# The component layer was removed by the user (a chart of listed companies only).
CURATED_COMPONENT_IDS: tuple = ()

RELATION_TYPES = ("SUPPLIES_TO", "INVESTED_IN", "PARTNERS_WITH")
COMPONENT_RELATION_TYPES = ("PRODUCES", "DEPENDS_ON")

# Domain knowledge used when live SEC ingestion discovers a company.
TICKER_TIERS = {
    # Tier -1: raw materials & specialty inputs
    "MP": -1, "UUUU": -1, "FCX": -1, "TECK": -1, "SCCO": -1,
    "CCJ": -1, "NXE": -1, "ENTG": -1, "LIN": -1, "PLAB": -1,
    "4063.T": -1, "3436.T": -1,
    "ASML": 0, "VRT": 0,
    # power generation, grid and fab equipment (Tier 0 bottlenecks)
    "VST": 0, "CEG": 0, "ETN": 0, "GEV": 0, "PWR": 0,
    "AMAT": 0, "LRCX": 0, "KLAC": 0,
    "TSM": 1, "MU": 1, "000660.KS": 1, "LITE": 1, "COHR": 1,
    "NVDA": 2, "AVGO": 2, "AMD": 2, "ANET": 2, "MRVL": 2, "INTC": 2,
    "ARM": 2, "QCOM": 2, "ALAB": 2, "CRDO": 2,
    "SMCI": 3, "DELL": 3, "HPE": 3,
    "MSFT": 4, "AMZN": 4, "GOOGL": 4, "META": 4, "AAPL": 4,
    "IREN": 4, "CRWV": 4, "ORCL": 4, "NBIS": 4,
}


def infer_tier(node_id: str, node_type: str = "Company") -> int:
    """Best-effort architectural tier for a node discovered at runtime."""
    key = re.sub(r"[^A-Z0-9.]", "", str(node_id).upper())
    return TICKER_TIERS.get(key, DEFAULT_TIER)


# Anything that can change the store. The text-to-Cypher studio and the agent's graph_query tool are
# both LLM-written, and "the prompt told it to stay read-only" is not a boundary: one hallucinated
# SET would edit the demo database the audience is looking at.
WRITE_KEYWORDS = ("CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
                  "ALTER", "GRANT", "LOAD", "CALL", "FOREACH")
_WRITE_KEYWORD = re.compile(r"\b(" + "|".join(WRITE_KEYWORDS) + r")\b", re.IGNORECASE)
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def read_only_violation(cypher: str) -> Optional[str]:
    """
    Return the write keyword found in a statement, or None if it is safe to run.

    Literals are blanked first: searching descriptions for the phrase "offset set" is a read, and a
    guard that refuses it teaches the studio's users to route around the guard.
    """
    match = _WRITE_KEYWORD.search(_QUOTED.sub("''", cypher or ""))
    return match.group(1).upper() if match else None


# `tier` is depth in the money flow; `layer` is the industrial segment. A reader asking "which
# layer of the photonics stack is single-source" needs the segment, and without it that question
# can only be answered by eyeballing the topology. Unknown tickers get no layer rather than a
# guessed one, so a layer filter never silently includes the wrong company.
LAYER_DEFINITIONS = {
    "materials": "Materials & Substrates",
    "eda": "EDA, IP & Design",
    "equipment": "Fab Equipment & Metrology",
    "foundry": "Foundry & Advanced Packaging",
    "memory": "Memory & HBM",
    "photonics": "Optics, Lasers & Photonics",
    "silicon": "Accelerators & Switch Silicon",
    "test": "Test & Inspection",
    "system": "Servers, Cooling & Facilities",
    "power": "Power, Grid & Fuel",
    "cloud": "Cloud & AI Demand",
}

TICKER_LAYERS = {
    "AXTI": "materials", "LWLG": "materials", "ENTG": "materials", "LIN": "materials",
    "PLAB": "materials", "4063.T": "materials", "3436.T": "materials", "MP": "materials",
    "FCX": "materials", "SCCO": "materials", "TECK": "materials",
    "SNPS": "eda", "CDNS": "eda", "ARM": "eda", "AIP": "eda",
    "ASML": "equipment", "AMAT": "equipment", "LRCX": "equipment", "KLAC": "equipment",
    "ONTO": "equipment",
    "TSM": "foundry", "INTC": "foundry", "GFS": "foundry", "UMC": "foundry", "TSEM": "foundry",
    "MU": "memory", "000660.KS": "memory", "SKHY": "memory", "WDC": "memory", "STX": "memory",
    "LITE": "photonics", "COHR": "photonics", "AAOI": "photonics", "POET": "photonics",
    "MTSI": "photonics", "LASR": "photonics", "CRDO": "photonics", "ALAB": "photonics",
    # The chart lists ST under Lasers, and its VCSEL / silicon-photonics line is why it belongs to
    # this graph at all, even though most of its revenue is general-purpose silicon.
    "STM": "photonics",
    "NVDA": "silicon", "AMD": "silicon", "AVGO": "silicon", "MRVL": "silicon",
    "ANET": "silicon", "QCOM": "silicon",
    "TER": "test", "AEHR": "test", "COHU": "test",
    "SMCI": "system", "DELL": "system", "HPE": "system", "VRT": "system", "AAON": "system",
    "VST": "power", "CEG": "power", "ETN": "power", "GEV": "power", "PWR": "power",
    "HUBB": "power", "POWL": "power", "OKLO": "power", "BWXT": "power", "CCJ": "power",
    "NXE": "power", "UUUU": "power",
    "MSFT": "cloud", "AMZN": "cloud", "GOOGL": "cloud", "META": "cloud", "AAPL": "cloud",
    "ORCL": "cloud", "CRWV": "cloud", "IREN": "cloud", "NBIS": "cloud", "PLTR": "cloud",
}


def infer_layer(node_id: str) -> Optional[str]:
    key = re.sub(r"[^A-Z0-9.]", "", str(node_id).upper())
    return LAYER_DEFINITIONS.get(TICKER_LAYERS.get(key, ""))


def _register_shutdown(func):
    """
    Runs `func` before ThreadPoolExecutor workers are joined.

    Plain `atexit` handlers run *after* concurrent.futures shuts its worker threads down,
    so closing the connection there is too late and the native destructor panics once the
    interpreter is gone. `threading._register_atexit` runs in the earlier phase.
    """
    try:
        import threading
        threading._register_atexit(func)  # noqa: SLF001 - documented shutdown phase hook
    except Exception:
        atexit.register(func)


class RelateStore:
    """
    Owns one Relate connection on a single worker thread and records every query.

    `relate.threadsafety == 1` means the connection cannot hop threads, and Streamlit
    reruns on new threads, so all access is funnelled through this executor.
    """

    MAX_HISTORY = 80

    def __init__(self, path: str):
        if not HAS_RELATE:
            raise RuntimeError(
                "relate-db is not installed. Install it from https://relatedb.ai "
                "(pip install relate-db) and restart the app."
            )
        self.path = path
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relate")
        self._conn = None
        # Connect on the worker thread so the connection is owned there for its whole life.
        self._executor.submit(self._open).result(timeout=60)
        _register_shutdown(self.close)

    # -- lifecycle ---------------------------------------------------------
    def _open(self):
        self._conn = relate.connect(self.path)
        logger.info("Connected to embedded Relate DB at: %s", self.path)

    def close(self):
        """Closes the native connection on its owning thread, then stops the executor."""
        closed = False
        try:
            self._executor.submit(self._close_conn).result(timeout=10)
            closed = True
        except Exception:
            pass
        if not closed and self._conn is not None:
            # Executor already torn down: best effort from this thread while Python lives.
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass

    def _close_conn(self):
        connection = self._conn
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self._conn = None

    # -- queries -----------------------------------------------------------
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Runs Cypher on the owning thread and returns rows as plain dicts."""
        return self._executor.submit(self._execute, cypher, params or {}).result()

    def _execute(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        import time

        started = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        error = None
        try:
            result = self._conn.execute(cypher, params)
            for row in result.fetchall():
                try:
                    rows.append(row.data() if hasattr(row, "data") else dict(row))
                except Exception:
                    rows.append({"_raw": str(row)})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                self._history.append({
                    "cypher": cypher.strip(),
                    "params": {k: v for k, v in params.items() if not isinstance(v, (list, set))},
                    "rows": len(rows),
                    "ms": round(elapsed, 1),
                    "error": error,
                })
        return rows

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)


class RelateSupplyChainGraph:
    """
    Graph API for the app, backed entirely by Relate.

    All public methods keep their previous signatures; the storage underneath is now
    Cypher instead of Python dictionaries.
    """

    def __init__(self, db_path: str = DEFAULT_RL_PATH):
        self.db_path = db_path
        self.chunks_file = DEFAULT_CHUNKS_PATH
        self.store = RelateStore(db_path)
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        # Compiled (name, node id, pattern) list, rebuilt whenever the node cache is reloaded.
        self._entity_index = None
        if self.count_nodes() == 0:
            self.seed_ai_supply_chain()
        self._ensure_chunk_schema()
        self._migrate_legacy_chunks()
        self.sync_layers()
        self.sync_tier_nodes()
        self._refresh_cache()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _rows(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.store.query(cypher, params)

    def _refresh_cache(self) -> None:
        rows = self._rows(
            "MATCH (n:Company) "
            "RETURN n.id AS id, n.label AS label, n.type AS type, n.tier AS tier, "
            "n.tier_name AS tier_name, n.full_name AS full_name, n.description AS description"
        )
        self._node_cache = {
            row["id"]: {
                "id": row["id"],
                "label": row.get("label") or row["id"],
                "type": row.get("type") or "Company",
                "tier": row["tier"] if row.get("tier") is not None else DEFAULT_TIER,
                "tier_name": row.get("tier_name")
                or TIER_DEFINITIONS.get(
                    row["tier"] if row.get("tier") is not None else DEFAULT_TIER, {}
                ).get("name", ""),
                "full_name": row.get("full_name"),
                "description": row.get("description"),
            }
            for row in rows
            if row.get("id")
        }
        self._entity_index = None

    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        """Derived index of Company nodes (Relate remains the source of truth)."""
        return self._node_cache

    @property
    def filings(self) -> Dict[str, Dict[str, Any]]:
        rows = self._rows(
            "MATCH (f:Filing) RETURN f.id AS id, f.ticker AS ticker, f.accession_no AS accession_no, "
            "f.form AS form, f.filing_date AS filing_date"
        )
        return {row["id"]: row for row in rows if row.get("id")}

    @property
    def edges(self) -> List[Dict[str, Any]]:
        """All company-to-company relationships, as dicts (derived from Relate)."""
        rows = self._rows(
            "MATCH (a:Company)-[r]->(b:Company) RETURN a.id AS source, b.id AS target, "
            "type(r) AS type, r.notes AS notes, r.source_form AS source_form, "
            "r.accession_no AS accession_no, r.as_of AS as_of, r.stake_pct AS stake_pct"
        )
        return [row for row in rows if row.get("source") and row.get("target")]

    def count_nodes(self) -> int:
        return self._rows("MATCH (n:Company) RETURN count(n) AS c")[0]["c"]

    def count_edges(self) -> int:
        rows = self._rows(
            "MATCH (a:Company)-[r]->(b:Company) RETURN count(r) AS c"
        )
        return rows[0]["c"] if rows else 0

    def count_filings(self) -> int:
        return self._rows("MATCH (f:Filing) RETURN count(f) AS c")[0]["c"]

    def recent_queries(self) -> List[Dict[str, Any]]:
        """Most recent Cypher statements, newest first (for the UI panel)."""
        return list(reversed(self.store.history()))

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).lower())

    def resolve_node_id(self, name: str) -> Optional[str]:
        """
        Maps a free-text entity name onto an existing node id.

        Exact id, then the curated alias table, then any spelling the graph itself carries. That is
        what lets "TSMC", "Taiwan Semiconductor" and "Taiwan Semiconductor (TSMC)" all land on the
        TSM node instead of being reported as an unknown company. The database is consulted last so
        a node written seconds ago, before the cache reloads, still resolves.
        """
        raw = (name or "").strip()
        if not raw:
            return None
        if raw in self._node_cache:
            return raw
        # Someone copying a label out of the graph view pastes "Taiwan Semiconductor (TSMC)", which
        # is two names and matches neither spelling on its own.
        variants = [raw] + [p.strip(" ,.&-'\"") for p in re.split(r"[(/,]", raw) if p.strip()]
        keys = {self._canonical_key(v) for v in variants}
        for variant in variants:
            target = KNOWN_ALIASES.get(variant.lower())
            if target and target in self._node_cache:
                return target
        for node_id in self._node_cache:
            if self._canonical_key(node_id) in keys:
                return node_id
        for candidate, node_id, _ in self._names():
            if self._canonical_key(candidate) in keys:
                return node_id
        row = self._rows("MATCH (n:Company {id: $id}) RETURN n.id AS id LIMIT 1", {"id": raw})
        return row[0]["id"] if row else None

    def _names(self):
        """
        Every spelling a question could use for a company, longest first, as (name, id, pattern).

        Two sources: the curated alias table and the names already stored on the nodes, so a ticker
        ingested next week resolves without anyone editing a table. Aliases whose target is not a
        node are skipped on purpose - "samsung" should read as unknown rather than quietly becoming
        a company that happens to be in the graph.
        """
        if self._entity_index is not None:
            return self._entity_index
        pairs: Dict[str, str] = {}

        def offer(raw_name, node_id):
            # "Taiwan Semiconductor (TSMC)" is one name in a filing and two in a question.
            for part in re.split(r"[(/]", str(raw_name or "")):
                name = part.strip(" ,.&-'\"").lower()
                if len(name) >= 3 and name not in pairs:
                    pairs[name] = node_id

        for node_id, info in self._node_cache.items():
            for raw_name in (node_id, info.get("label"), info.get("full_name")):
                offer(raw_name, node_id)
        for alias, node_id in KNOWN_ALIASES.items():
            if node_id in self._node_cache:
                offer(alias, node_id)

        self._entity_index = [
            (name, node_id, re.compile(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])"))
            for name, node_id in sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0]))
        ]
        return self._entity_index

    def find_entities(self, text: str) -> List[str]:
        """
        Node ids for companies named in free text, ordered by where they first appear.

        Capitalisation is used as a precision rule, not a formality: the alias table deliberately
        contains plain English words ("meta", "coherent", "azure"), so a question that capitalises
        its proper nouns must not treat those words as companies mid-sentence. A question typed
        entirely in lowercase gets no such filter, because then there is no signal to read.
        """
        if not text:
            return []
        lowered = text.lower()
        permissive = lowered == text
        positions: Dict[str, int] = {}
        for name, node_id, pattern in self._names():
            match = pattern.search(lowered)
            if not match:
                continue
            if not permissive and not text[match.start()].isupper():
                continue
            current = positions.get(node_id)
            positions[node_id] = match.start() if current is None else min(current, match.start())
        return [nid for nid, _ in sorted(positions.items(), key=lambda kv: kv[1])]

    def unindexed_entities(self, text: str) -> List[str]:
        """
        Names the question capitalises that match no node.

        This is the difference between answering "which companies depend on Corning for glass" with
        Micron and Coherent excerpts, and saying Corning has never been ingested. Word runs that
        resolve through find_entities are dropped, as are interrogatives and section labels that
        merely happen to start a sentence.
        """
        out: List[str] = []
        for span in re.findall(r"[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3}", text or ""):
            name = span.strip(" -,.")
            if len(name) < 4 or name.lower() in NAME_STOPWORDS or name in out:
                continue
            if self.resolve_node_id(name) or self.find_entities(name):
                continue
            out.append(name)
        return out

    def get_component_id(self, name: str) -> Optional[str]:
        """Component nodes are no longer part of the graph, so this always returns None."""
        return None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_node(self, node_id: str, label: str, node_type: str = "Company",
                 tier: Optional[int] = None, **props):
        """
        Idempotently MERGEs a node. `tier=None` means "infer it"; an existing node keeps
        its tier unless an explicit tier is passed.
        """
        if not node_id:
            return
        existing = self.resolve_node_id(node_id)
        if existing and existing != node_id:
            node_id = existing  # canonical id wins

        resolved_tier = infer_tier(node_id, node_type) if tier is None else tier
        properties = {
            "label": label or node_id,
            "type": node_type,
            "tier": resolved_tier,
            "tier_name": TIER_DEFINITIONS.get(resolved_tier, {}).get("name", f"Tier {resolved_tier}"),
        }
        layer = infer_layer(node_id) or props.get("layer")
        if layer:
            properties["layer"] = layer
        for key in ("full_name", "description", "notes"):
            if props.get(key):
                properties[key] = props[key]

        if existing:
            if tier is None:
                return  # keep the stored tier unchanged
            self._rows(
                "MATCH (n:Company {id: $id}) SET n.tier = $tier, n.tier_name = $tier_name",
                {"id": node_id, "tier": resolved_tier, "tier_name": properties["tier_name"]},
            )
            self._refresh_cache()
            return

        self._rows(
            "MERGE (n:Company {id: $id}) "
            "SET n.label = $label, n.type = $type, n.tier = $tier, n.tier_name = $tier_name, "
            "n.full_name = $full_name, n.description = $description, n.layer = $layer",
            {
                "id": node_id,
                "label": properties["label"],
                "type": node_type,
                "tier": resolved_tier,
                "tier_name": properties["tier_name"],
                "full_name": props.get("full_name"),
                "description": props.get("description"),
                # An unmapped ticker is written as NULL, which reads as "not in any layer" rather
                # than being quietly filtered out of a layer question later.
                "layer": properties.get("layer"),
            },
        )
        self._refresh_cache()

    def add_edge(self, source: str, target: str, rel_type: str, **props) -> bool:
        """
        Idempotently MERGEs a relationship with provenance. Endpoints are resolved against
        existing node ids, so an extracted 'COWOS'-style casing slip cannot dangle.
        """
        if rel_type not in RELATION_TYPES:
            return False
        source_id = self.resolve_node_id(source)
        target_id = self.resolve_node_id(target)
        if not source_id or not target_id or source_id == target_id:
            return False

        exists = self._rows(
            f"MATCH (a:Company {{id: $src}})-[r:{rel_type}]->(b:Company {{id: $tgt}}) "
            "RETURN r AS rel LIMIT 1",
            {"src": source_id, "tgt": target_id},
        )
        if exists:
            return False

        edge_props = {}
        for key in ("notes", "source_form", "accession_no", "as_of", "stake_pct", "evidence"):
            if props.get(key) is not None:
                edge_props[key] = props[key]

        set_clause = ""
        params = {"src": source_id, "tgt": target_id}
        if edge_props:
            assignments = []
            for key, value in edge_props.items():
                assignments.append(f"r.{key} = ${key}")
                params[key] = value
            set_clause = " SET " + ", ".join(assignments)

        self._rows(
            f"MATCH (a:Company {{id: $src}}), (b:Company {{id: $tgt}}) "
            f"MERGE (a)-[r:{rel_type}]->(b){set_clause}",
            params,
        )
        return True

    def record_filing(self, filing_id: str, ticker: str, accession_no: str,
                      form_type: str, filing_date: str):
        """Records an ingested filing as a graph node linked to its company."""
        self._rows(
            "MERGE (c:Company {id: $ticker}) "
            "ON CREATE SET c.label = $ticker, c.type = 'Company', c.tier = $tier, "
            "c.tier_name = $tier_name, c.layer = $layer "
            "MERGE (f:Filing {id: $id}) "
            "SET f.ticker = $ticker, f.accession_no = $accession, f.form = $form, f.filing_date = $date "
            "MERGE (c)-[:FILED]->(f)",
            {
                "ticker": ticker,
                "id": filing_id,
                "accession": accession_no,
                "form": form_type,
                "date": filing_date,
                "tier": infer_tier(ticker),
                "tier_name": TIER_DEFINITIONS.get(infer_tier(ticker), {}).get("name", ""),
                "layer": infer_layer(ticker),
            },
        )

    def check_filing_exists(self, filing_id: str) -> bool:
        return bool(self._rows("MATCH (f:Filing {id: $id}) RETURN f.id AS id LIMIT 1", {"id": filing_id}))

    def _repair_edges(self) -> bool:
        """No-op: Cypher MATCH-edges cannot dangle, so there is nothing to repair."""
        return False

    def infer_component_tiers(self) -> int:
        """No-op: the component layer has been retired, so there is nothing to re-tier."""
        return 0

    def prune_uncurated_components(self) -> List[str]:
        """Deletes every component node (the layer is retired), together with its edges."""
        removed = [row["id"] for row in self._rows("MATCH (n:Component) RETURN n.id AS id")]
        if removed:
            self._rows("MATCH (n:Component) DETACH DELETE n")
            logger.warning("Pruned %d component node(s): %s", len(removed), removed)
        self._refresh_cache()
        return removed

    # ------------------------------------------------------------------
    # Chunks - filing excerpts and their vectors live in Relate, indexed natively
    # ------------------------------------------------------------------
    def _ensure_chunk_schema(self) -> None:
        """Create the vector and full-text indexes over :Chunk (idempotent)."""
        for statement in (
            "CREATE VECTOR INDEX chunk_emb IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            "WITH [c.ticker] OPTIONS {indexConfig: {"
            "`vector.dimensions`: " + str(CHUNK_EMBEDDING_DIM) + ", "
            "`vector.similarity_function`: 'cosine'}}",
            "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
        ):
            try:
                self._rows(statement)
            except Exception as exc:
                logger.warning("Could not create a chunk index: %s", exc)

    def _migrate_legacy_chunks(self) -> int:
        """
        Move the old supply_chain_chunks.json sidecar into Relate once, and leave the file in
        place. The sidecar path is global while stores are per-path, so a scratch database can
        open it; renaming it as a side effect would remove the only copy of chunks that store
        never received. Re-running is cheap - the skip below and MERGE on chunk id make it
        idempotent - so the file is kept until it is deliberately retired.
        """
        if not os.path.exists(self.chunks_file):
            return 0
        try:
            with open(self.chunks_file, "r", encoding="utf-8") as handle:
                legacy = json.load(handle)
        except Exception as exc:
            logger.warning("Could not read legacy chunks: %s", exc)
            return 0
        if not legacy:
            return 0

        already = self.chunk_count()
        if already >= len(legacy):
            logger.info("Chunks already live in Relate (%d); sidecar left untouched.", already)
            return 0

        written = self.save_chunks(legacy)
        if self.chunk_count() < len(legacy):
            logger.warning("Migration incomplete (%d of %d chunks in Relate); keeping %s",
                           self.chunk_count(), len(legacy), self.chunks_file)
            return written
        logger.info("Migrated %d chunks into Relate; %s left in place as a backup.",
                    written, self.chunks_file)
        return written

    def sync_layers(self) -> int:
        """
        Tag every tracked company with its industrial layer.

        Nodes predating the taxonomy carry no layer, and a missing property reads as 'not in
        any layer' to a Cypher filter, so this runs on every start rather than only on seeding.
        """
        rows = [{"id": t, "layer": name} for t, key in TICKER_LAYERS.items()
                for name in [LAYER_DEFINITIONS.get(key)] if name]
        if not rows:
            return 0
        self._rows(
            "UNWIND $rows AS r MATCH (n:Company {id: r.id}) SET n.layer = r.layer",
            {"rows": rows},
        )
        return len(rows)

    def sync_tier_nodes(self) -> int:
        """
        Give the tier taxonomy a place in the graph: (:Company)-[:IN_TIER]->(:Tier) and
        (:Tier)-[:FEEDS]->(:Tier).

        Until now a tier was an integer on a company, so "what sits between the materials layer and
        the hyperscalers" could only be answered by Python aggregation - the spine existed in the
        code, not in the database, and a query written against the graph could not see it. Only
        structure is stored here. Edge counts and citation coverage are derived on read by
        get_tier_summaries, because a stored count starts lying the moment a filing is ingested.
        """
        owned = {row["id"]: row["tier"] for row in self._rows(
            "MATCH (c:Company) WHERE c.tier IS NOT NULL RETURN c.id AS id, c.tier AS tier"
        )}
        stored = {row["c"]: row["t"] for row in self._rows(
            "MATCH (c:Company)-[:IN_TIER]->(t:Tier) RETURN c.id AS c, t.tier AS t"
        )}
        have_tiers = {row["t"] for row in self._rows("MATCH (t:Tier) RETURN t.tier AS t")}
        desired_flows = self._rows(
            "MATCH (a:Company)-[e]->(b:Company) "
            "WHERE a.tier IS NOT NULL AND b.tier IS NOT NULL AND a.tier <> b.tier "
            "RETURN a.tier AS src, b.tier AS dst"
        )
        stored_flows = self._rows("MATCH (s:Tier)-[:FEEDS]->(d:Tier) RETURN s.tier AS src, d.tier AS dst")
        if (owned == stored and have_tiers == set(TIER_DEFINITIONS)
                and {(r["src"], r["dst"]) for r in desired_flows}
                == {(r["src"], r["dst"]) for r in stored_flows}):
            return 0

        self._rows(
            "UNWIND $rows AS r MERGE (t:Tier {id: r.id}) "
            "SET t.tier = r.tier, t.name = r.name, t.short = r.short, t.color = r.color, "
            "t.edge_color = r.edge_color",
            {"rows": [{"id": tier_node_id(tier), "tier": tier, "name": meta["name"],
                       "short": meta["short"], "color": meta["color"],
                       "edge_color": meta["edge_color"]} for tier, meta in TIER_DEFINITIONS.items()]},
        )
        # A company that moved tiers keeps an edge to the tier it left unless the stale pair goes first.
        moved = [cid for cid, tier in stored.items() if owned.get(cid) != tier]
        if moved:
            self._rows("UNWIND $ids AS i MATCH (c:Company {id: i})-[r:IN_TIER]->(:Tier) DELETE r",
                       {"ids": moved})
        self._rows(
            "UNWIND $rows AS r MATCH (c:Company {id: r.id}) MATCH (t:Tier {tier: r.tier}) "
            "MERGE (c)-[:IN_TIER]->(t)",
            {"rows": [{"id": cid, "tier": tier} for cid, tier in owned.items()]},
        )
        self._rows("MATCH (s:Tier)-[f:FEEDS]->(d:Tier) DELETE f")
        pairs = sorted({(r["src"], r["dst"]) for r in desired_flows})
        if pairs:
            self._rows(
                "UNWIND $rows AS r MATCH (s:Tier {tier: r.src}) MATCH (d:Tier {tier: r.dst}) "
                "MERGE (s)-[:FEEDS]->(d)",
                {"rows": [{"src": src, "dst": dst} for src, dst in pairs]},
            )
        logger.info("Tier spine materialized: %d tiers, %d members, %d flows.",
                    len(TIER_DEFINITIONS), len(owned), len(pairs))
        return len(owned)

    def chunk_count(self) -> int:
        return self._rows("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Batch-write excerpts as (:Chunk)-[:ABOUT]->(:Company) in one statement, so a filing of
        N excerpts costs one round trip instead of N full-file rewrites.
        """
        rows = []
        for chunk in chunks or []:
            vector = chunk.get("embedding") or []
            if not chunk.get("id") or not chunk.get("text"):
                continue
            if len(vector) != CHUNK_EMBEDDING_DIM:
                logger.warning("Skipping chunk %s: %d-dim vector, index expects %d",
                               chunk.get("id"), len(vector), CHUNK_EMBEDDING_DIM)
                continue
            rows.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "ticker": (chunk.get("ticker") or "").upper(),
                "accession_no": chunk.get("accession_no"),
                "form": chunk.get("form"),
                "as_of": chunk.get("as_of"),
                "heading": chunk.get("heading") or "",
                "fp": chunk.get("fp") or hashlib.sha1(
                    re.sub(r"\W+", " ", chunk["text"].lower()).strip().encode("utf-8")
                ).hexdigest()[:16],
                "e": [float(v) for v in vector],
            })
        if not rows:
            return 0
        # An excerpt whose ticker has no Company node can never join the graph, which would
        # make it plain RAG. Seed the node from curated tier metadata (the ticker is proven by
        # the filing itself); no relationship is implied.
        for ticker in {row["ticker"] for row in rows if row["ticker"]}:
            if not self.resolve_node_id(ticker):
                self.add_node(ticker, ticker, "Company")  # add_node refreshes the cache
        self._rows(
            "UNWIND $rows AS r "
            "MERGE (c:Chunk {id: r.id}) "
            "SET c.text = r.text, c.ticker = r.ticker, c.accession_no = r.accession_no, "
            "c.form = r.form, c.as_of = r.as_of, c.heading = r.heading, c.fp = r.fp, "
            "c.embedding = vector(r.e, $dim, FLOAT32) "
            "WITH c MATCH (co:Company {id: c.ticker}) MERGE (c)-[:ABOUT]->(co)",
            {"rows": rows, "dim": CHUNK_EMBEDDING_DIM},
        )
        return len(rows)

    def chunk_fingerprints(self, ticker: str) -> set:
        """Text fingerprints already stored for a ticker, so re-ingest adds only what is new."""
        rows = self._rows("MATCH (c:Chunk {ticker: $t}) RETURN c.fp AS fp", {"t": (ticker or "").upper()})
        return {row["fp"] for row in rows if row.get("fp")}

    @staticmethod
    def _fulltext_query(query_text: str) -> str:
        """
        Reduce a user question to plain terms. Relate's full-text parser reads ^ as a boost,
        ~ as fuzzy, * and ? as wildcards and uppercase AND/OR/NOT as operators, so those have
        to go or a question about "co-owned" stock silently changes meaning.
        """
        cleaned = re.sub(r"[^A-Za-z0-9' ]+", " ", query_text or "")
        cleaned = re.sub(r"\b(AND|OR|NOT|TO)\b", " ", cleaned)
        return " ".join(cleaned.split())[:200]

    def hybrid_search_chunks(self, query_text: str, query_embedding: Optional[List[float]] = None,
                             top_k: int = 5,
                             tickers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        One statement: Relate's vector index and BM25 full-text index are fused with
        Reciprocal Rank Fusion, then each hit hops to the Company it describes.

        `tickers` adds a third list: the named companies' own filings, ranked by the same cosine
        scores. It is a boost rather than a filter, so a supplier's disclosure about its customer
        still competes - it just stops winning by default.

        A chunk in both lists outranks a chunk leading one, which is what keeps an exact
        accession-number or product-name match from being buried by generic prose.
        Without an embedding the BM25 arm is used on its own instead of degrading to a
        keyword count - and with no cosine to judge, no relevance gate is applied.
        """
        limit = max(1, min(int(top_k or 5), 10))
        params = {"q": self._fulltext_query(query_text), "k": limit, "rrf": RRF_K}
        usable_vector = (
            query_embedding is not None and len(query_embedding) == CHUNK_EMBEDDING_DIM
        )
        if not usable_vector:
            if not params["q"]:
                return []
            statement = """
MATCH (c:Chunk)
SEARCH c IN (FULLTEXT INDEX chunk_text FOR $q LIMIT $k) SCORE AS s
WITH c, s ORDER BY s DESC, c.id
WITH collect(c) AS cs
UNWIND range(0, size(cs) - 1) AS i
WITH cs[i] AS c, i + 1 AS rank
ORDER BY rank LIMIT $k
OPTIONAL MATCH (c)-[:ABOUT]->(co:Company)
RETURN c.id AS id, c.text AS text, c.ticker AS ticker, c.accession_no AS accession_no,
       c.form AS form, c.as_of AS as_of, c.heading AS heading,
       1.0 / ($rrf + rank) AS rrf, 1 AS lists, null AS v_rank, rank AS f_rank, collect(co.tier) AS tiers
"""
        else:
            params["v"] = [float(value) for value in query_embedding]
            params["min_sim"] = MIN_VECTOR_SIM
            # Both vector arms carry cosine, so one threshold covers them. Lexical agreement is not
            # allowed to stand alone: "how do I reset my password" pulls Oracle excerpts at BM25 4.4
            # that describe nothing relevant, and that is exactly the text a model will quote back.
            scope = [t for t in (tickers or []) if t][:2]
            statement = """
CALL () {
  MATCH (c:Chunk)
  SEARCH c IN (VECTOR INDEX chunk_emb FOR $v LIMIT $k) SCORE AS s
  WITH c, s ORDER BY s DESC, c.id
  WITH collect(c) AS cs, collect(s) AS ss
  UNWIND range(0, size(cs) - 1) AS i
  RETURN cs[i] AS c, 'vector' AS list, i + 1 AS rank, ss[i] AS score
  UNION ALL
  MATCH (c:Chunk)
  SEARCH c IN (FULLTEXT INDEX chunk_text FOR $q LIMIT $k) SCORE AS s
  WITH c, s ORDER BY s DESC, c.id
  WITH collect(c) AS cs, collect(s) AS ss
  UNWIND range(0, size(cs) - 1) AS i
  RETURN cs[i] AS c, 'fulltext' AS list, i + 1 AS rank, ss[i] AS score"""
            if scope:
                params["scope"] = scope
                # Widening before filtering is what makes this a search of that company's filings
                # rather than an intersection with five globally nearest chunks.
                statement += """
  UNION ALL
  MATCH (c:Chunk)
  SEARCH c IN (VECTOR INDEX chunk_emb FOR $v LIMIT $wide) SCORE AS s
  WITH c, s WHERE c.ticker IN $scope
  WITH c, s ORDER BY s DESC, c.id
  WITH collect(c) AS cs, collect(s) AS ss
  UNWIND range(0, size(cs) - 1) AS i
  RETURN cs[i] AS c, 'scoped' AS list, i + 1 AS rank, ss[i] AS score"""
                params["wide"] = min(200, max(60, limit * 20))
            statement += """
}
WITH c, sum(1.0 / ($rrf + rank)) AS rrf, count(*) AS lists,
     min(CASE list WHEN 'vector' THEN rank END) AS v_rank,
     min(CASE list WHEN 'fulltext' THEN rank END) AS f_rank,
     max(CASE WHEN list <> 'fulltext' THEN score END) AS v_score
WHERE v_score IS NOT NULL AND v_score >= $min_sim
OPTIONAL MATCH (c)-[:ABOUT]->(co:Company)
WITH c, rrf, lists, v_rank, f_rank, v_score, collect(co.tier) AS tiers
// RRF scores tie constantly at small k, and an alphabetical tie-break would let an unrelated
// semantic near-match outrank a chunk that literally contains the accession number being
// asked for - so ties go to breadth, then to the lexical (full-text) hit.
ORDER BY rrf DESC, lists DESC, coalesce(f_rank, 999) ASC, coalesce(v_rank, 999) ASC, c.id
LIMIT $k
RETURN c.id AS id, c.text AS text, c.ticker AS ticker, c.accession_no AS accession_no,
       c.form AS form, c.as_of AS as_of, c.heading AS heading,
       rrf, lists, v_rank, f_rank, round(v_score, 4) AS v_score, tiers
"""
        if not self.chunk_count():
            return []
        try:
            return self._rows(statement, params)
        except Exception as exc:
            logger.warning("Fused search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Seed + export
    # ------------------------------------------------------------------
    def seed_ai_supply_chain(self):
        """
        Pre-seeds the complete 'AI Money Cycle' into Relate with architectural tiers.
        Listed companies only: no component/product nodes.
        """
        logger.info("Pre-seeding Tiered AI Supply Chain Knowledge Graph in Relate...")

        companies = [
            ("ASML", "ASML Holding", "EUV Lithography Systems", 0),
            ("VRT", "Vertiv Holdings", "Liquid Cooling & Power Infrastructure", 0),
            ("TSM", "Taiwan Semiconductor (TSMC)", "Pure-play Foundry & CoWoS Packaging", 1),
            ("MU", "Micron Technology", "High Bandwidth Memory (HBM3e)", 1),
            ("000660.KS", "SK Hynix", "High Bandwidth Memory (HBM3e)", 1),
            ("NVDA", "NVIDIA", "AI Superchips (Hopper & Blackwell)", 2),
            ("AVGO", "Broadcom", "Custom AI ASICs & Networking Silicon", 2),
            ("AMD", "Advanced Micro Devices", "AI Accelerators (MI300X/MI325X)", 2),
            ("ANET", "Arista Networks", "AI Data Center Switching & Optics", 2),
            ("SMCI", "Super Micro Computer", "AI Server Integrator & Direct Liquid Cooling", 3),
            ("MSFT", "Microsoft", "Azure AI Cloud & Copilot", 4),
            ("AMZN", "Amazon AWS", "AWS Cloud Infrastructure & Trainium", 4),
            ("GOOGL", "Alphabet / Google", "Google Cloud & TPU Infrastructure", 4),
            ("META", "Meta Platforms", "Meta AI Infrastructure & Llama", 4),
            ("AAPL", "Apple", "Apple Intelligence Private Cloud Compute", 4),
        ]
        for ticker, name, desc, tier in companies:
            self.add_node(ticker, ticker, "Company", tier=tier, full_name=name, description=desc)

        materials = [
            ("MP", "MP Materials", "Rare-earth mining & NdPr magnet inputs"),
            ("UUUU", "Energy Fuels", "Uranium & rare-earth processing"),
            ("FCX", "Freeport-McMoRan", "Copper (power, wiring, cold plates)"),
            ("TECK", "Teck Resources", "Copper & germanium byproduct"),
            ("SCCO", "Southern Copper", "Copper concentrate & refined copper"),
            ("CCJ", "Cameco", "Uranium fuel for nuclear generation"),
            ("NXE", "NexGen Energy", "Uranium development"),
            ("ENTG", "Entegris", "Fab materials, filtration & contamination control"),
            ("LIN", "Linde", "Industrial gases (nitrogen, helium, neon)"),
            ("PLAB", "Photronics", "Photomasks for wafer lithography"),
            ("4063.T", "Shin-Etsu Chemical", "Silicon wafers, photoresist, quartz"),
            ("3436.T", "SUMCO", "Silicon wafers"),
        ]
        for ticker, name, desc in materials:
            self.add_node(ticker, ticker, "Company", tier=-1, full_name=name, description=desc)

        supply_edges = [
            ("ASML", "TSM", "EUV Lithography Systems"),
            ("VRT", "SMCI", "Liquid Cooling Distribution Units"),
            ("VRT", "MSFT", "Hyperscale Thermal Management"),
            ("TSM", "NVDA", "3nm/4nm Wafer Fab & CoWoS Packaging"),
            ("TSM", "AMD", "Wafer Fabrication & 2.5D Packaging"),
            ("TSM", "AVGO", "Custom Silicon Fabrication"),
            ("MU", "NVDA", "HBM3e Memory Stacks"),
            ("000660.KS", "NVDA", "HBM3e Memory Stacks"),
            ("NVDA", "SMCI", "HGX B200 / H100 Motherboards"),
            ("NVDA", "MSFT", "Blackwell & Hopper Superclusters"),
            ("NVDA", "AMZN", "DGX Cloud & UltraClusters"),
            ("NVDA", "GOOGL", "AI Supercomputing Clusters"),
            ("NVDA", "META", "Large AI Training Clusters"),
            ("AVGO", "GOOGL", "Custom TPU Co-Design & SerDes"),
            ("AVGO", "META", "MTIA Custom AI Silicon"),
            ("ANET", "MSFT", "400G/800G AI Back-end Switching"),
            ("ANET", "META", "AI Cluster Fabric"),
            ("SMCI", "MSFT", "Turnkey AI Server Racks"),
            ("SMCI", "AMZN", "DLC Server Racks"),
            ("NVDA", "DELL", "AI server integration (curated baseline, same class as SMCI)"),
            ("NVDA", "HPE", "AI server integration (curated baseline, same class as SMCI)"),
            ("TSM", "AAPL", "A18 Pro & M4 Advanced Silicon Fab"),
            ("NVDA", "AAPL", "AI Server Clusters for Apple Intelligence"),
            # Tier -1 materials into the stack (curated baseline: materials vendors do not
            # name their fab customers in filings)
            ("CCJ", "CEG", "Uranium fuel supply for nuclear generation"),
            ("LIN", "TSM", "Industrial gases to wafer fabs"),
            ("ENTG", "TSM", "Fab materials, filtration & contamination control"),
            ("PLAB", "TSM", "Photomasks for wafer lithography"),
            ("4063.T", "TSM", "Silicon wafers, photoresist & quartz"),
            ("3436.T", "TSM", "Silicon wafers"),
            ("TECK", "COHR", "Germanium for optical components"),
        ]
        for source, target, notes in supply_edges:
            self.add_edge(source, target, "SUPPLIES_TO", notes=notes)

        self._refresh_cache()
        logger.info("Seeding complete: %d nodes, %d edges.", self.count_nodes(), self.count_edges())

    def export_json(self, path: Optional[str] = None):
        """Writes a JSON snapshot of the graph (backup/debug only; never read as truth)."""
        target = path or self.db_path.replace(".rl", ".json")
        try:
            nodes = self._rows(
                "MATCH (n:Company) RETURN n.id AS id, n.label AS label, n.type AS type, "
                "n.tier AS tier, n.tier_name AS tier_name, n.layer AS layer, "
                "n.full_name AS full_name, n.description AS description"
            )
            edges = self._rows(
                "MATCH (a:Company)-[r]->(b:Company) RETURN a.id AS source, b.id AS target, "
                "type(r) AS type, r.notes AS notes, r.source_form AS source_form, "
                "r.accession_no AS accession_no, r.as_of AS as_of, r.stake_pct AS stake_pct, "
                "r.evidence AS evidence"
            )
            payload = {
                "nodes": {row["id"]: row for row in nodes},
                "edges": edges,
                "filings": self.filings,
            }
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Could not export graph JSON: %s", exc)

    # Backwards-compatible name used by ingestion: refresh derived structure, then snapshot it.
    def _persist(self):
        self.sync_tier_nodes()
        self.export_json()

    def import_from_json(self, path: str) -> Dict[str, int]:
        """
        One-time migration: loads a JSON snapshot into Relate.

        Kept in the product so a demo DB can be rebuilt from a committed snapshot, but the
        app never reads JSON as its source of truth.
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        filings = data.get("filings", {})

        imported_nodes = 0
        for node_id, info in nodes.items():
            node_type = info.get("type", "Company")
            tier = info.get("tier")
            self._rows(
                "MERGE (n:Company {id: $id}) SET n.label = $label, n.type = $type, n.tier = $tier, "
                "n.tier_name = $tier_name, n.full_name = $full_name, n.description = $description, "
                "n.layer = $layer",
                {
                    "id": node_id,
                    "label": info.get("label") or node_id,
                    "type": node_type,
                    "tier": tier if tier is not None else infer_tier(node_id, node_type),
                    "tier_name": info.get("tier_name")
                    or TIER_DEFINITIONS.get(tier if tier is not None else DEFAULT_TIER, {}).get("name", ""),
                    "full_name": info.get("full_name"),
                    "description": info.get("description"),
                    "layer": info.get("layer") or infer_layer(node_id),
                },
            )
            imported_nodes += 1

        imported_edges = 0
        for edge in edges:
            rel = edge.get("type")
            if rel not in RELATION_TYPES:
                continue
            props = {key: edge.get(key) for key in
                     ("notes", "source_form", "accession_no", "as_of", "stake_pct", "evidence")
                     if edge.get(key) is not None}
            set_clause = ""
            params = {"src": edge["source"], "tgt": edge["target"]}
            if props:
                set_clause = " SET " + ", ".join(f"r.{key} = ${key}" for key in props)
                params.update(props)
            self._rows(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) MERGE (a)-[r:{rel}]->(b){set_clause}",
                params,
            )
            imported_edges += 1

        for filing_id, info in filings.items():
            self.record_filing(filing_id, info.get("ticker", ""), info.get("accession_no", ""),
                               info.get("form", ""), info.get("filing_date", ""))

        self.sync_layers()
        self.sync_tier_nodes()
        self._refresh_cache()
        return {"nodes": imported_nodes, "edges": imported_edges, "filings": len(filings)}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_tiered_stack(
        self,
        selected_tiers: Optional[List[int]] = None,
        show_components: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, int]]:
        """Tiered strata for the AI Stack view, read from Relate with Cypher."""
        tiers = selected_tiers if selected_tiers is not None else [-1, 0, 1, 2, 3, 4]
        if not tiers:
            return {}, [], {}

        node_rows = self._rows(
            "MATCH (n:Company) WHERE n.tier IN $tiers "
            "RETURN n.id AS id, n.label AS label, n.tier AS tier, n.tier_name AS tier_name, "
            "n.full_name AS full_name, n.description AS description",
            {"tiers": list(tiers)},
        )
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        tier_counts = {tier: 0 for tier in tiers}
        for row in node_rows:
            tier = row["tier"] if row.get("tier") is not None else DEFAULT_TIER
            tier_meta = TIER_DEFINITIONS.get(tier, {"color": "#757575"})
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            nodes_dict[row["id"]] = {
                "id": row["id"],
                "label": row.get("label") or row["id"],
                "color": tier_meta["color"],
                "size": 28,
                "tier": tier,
                "type": "Company",
                "kind": "company",
                "shape": "box",
                "title": f"<b>{row.get('full_name') or row['id']}</b><br>{row.get('tier_name', '')}"
                         + (f"<br>{row['description']}" if row.get("description") else ""),
            }

        edge_rows = self._rows(
            "MATCH (a:Company)-[r]->(b:Company) "
            "WHERE a.tier IN $tiers AND b.tier IN $tiers "
            "AND type(r) IN ['SUPPLIES_TO', 'INVESTED_IN', 'PARTNERS_WITH'] "
            "RETURN a.id AS source, b.id AS target, type(r) AS rel, r.notes AS notes, "
            "r.source_form AS source_form, r.accession_no AS accession_no, r.as_of AS as_of",
            {"tiers": list(tiers)},
        )
        edges_list = []
        for row in edge_rows:
            source_tier = nodes_dict.get(row["source"], {}).get("tier", DEFAULT_TIER)
            color = RELATION_EDGE_COLORS.get(
                row["rel"], TIER_DEFINITIONS.get(source_tier, {}).get("edge_color", "#9E9E9E")
            )
            edges_list.append({
                "source": row["source"],
                "target": row["target"],
                "type": row["rel"],
                "color": color,
                "notes": row.get("notes") or "",
                "source_form": row.get("source_form"),
                "accession_no": row.get("accession_no"),
                "as_of": row.get("as_of"),
            })

        return nodes_dict, edges_list, tier_counts

    def get_layer_map(self) -> List[Dict[str, Any]]:
        """
        Per-layer concentration, the question a tier view cannot answer.

        A tier says how deep in the money flow a company sits; this says whether the layer it
        sits in is a chokepoint - how many buyers its members serve, and how much of that load
        one company carries. Evidence coverage travels with it, because a layer that is dense
        but uncited is a claim, not a finding.
        """
        members = {
            row["layer"]: row["n"] for row in self._rows(
                "MATCH (n:Company) WHERE n.layer IS NOT NULL "
                "RETURN n.layer AS layer, count(n) AS n"
            )
        }
        flows = self._rows(
            "MATCH (a:Company)-[r:SUPPLIES_TO]->(b:Company) WHERE a.layer IS NOT NULL "
            "RETURN a.layer AS layer, a.id AS supplier, count(DISTINCT b.id) AS buyers"
        )
        citations = {
            row["layer"]: row for row in self._rows(
                "MATCH (a:Company)-[r]->(b:Company) WHERE a.layer IS NOT NULL "
                "RETURN a.layer AS layer, count(r) AS edges, "
                "sum(CASE WHEN r.accession_no IS NULL THEN 1 ELSE 0 END) AS uncited"
            )
        }
        by_layer: Dict[str, List[Dict[str, Any]]] = {}
        for row in flows:
            by_layer.setdefault(row["layer"], []).append(row)

        out: List[Dict[str, Any]] = []
        for layer, supplier_rows in by_layer.items():
            supplier_rows.sort(key=lambda r: -r["buyers"])
            total = sum(r["buyers"] for r in supplier_rows)
            top = supplier_rows[0]
            cite = citations.get(layer, {})
            out.append({
                "layer": layer,
                "members": members.get(layer, 0),
                "suppliers_with_edges": len(supplier_rows),
                "supply_edges": total,
                "top_supplier": top["supplier"],
                "top_supplier_buyers": top["buyers"],
                # Share of the layer's buyer relationships carried by its largest supplier: the
                # single number that separates a competitive layer from a single point of failure.
                "load_share": round(top["buyers"] / total, 2) if total else 0.0,
                "edges": cite.get("edges", 0),
                "uncited_edges": cite.get("uncited", 0),
                "evidence_pct": round(100.0 * (cite.get("edges", 0) - cite.get("uncited", 0))
                                      / cite["edges"], 1) if cite.get("edges") else 0.0,
            })
        out.sort(key=lambda item: (-item["top_supplier_buyers"], -item["load_share"]))
        return out

    def get_tier_summaries(self) -> List[Dict[str, Any]]:
        """
        Per-tier rollup read off the tier spine: who belongs to a stratum, what it feeds, and how
        much of that flow a filing actually supports.

        Every number here is aggregated live from the company edges rather than stored on the tier
        node, so ingesting a filing cannot leave a stale count behind. `evidence_pct` counts only
        outgoing edges: a supplier's own disclosure is what names the relationship, so a buyer's
        silence about a vendor does not count against the vendor's tier.
        """
        rows = {row["tier"]: dict(row, feeds=[], fed_by=[], out_edges=0, out_uncited=0,
                                  in_edges=0, in_uncited=0)
                for row in self._rows(
                    "MATCH (t:Tier) RETURN t.tier AS tier, t.name AS name, t.short AS short, "
                    "t.color AS color, t.edge_color AS edge_color"
                )}
        for row in self._rows(
            "MATCH (t:Tier)<-[:IN_TIER]-(c:Company) "
            "RETURN t.tier AS tier, count(c) AS members, collect(c.id) AS tickers"
        ):
            tier = rows.get(row["tier"])
            if tier:
                tier["members"] = row["members"]
                tier["tickers"] = sorted(row["tickers"] or [])
        for row in self._rows(
            "MATCH (s:Tier)-[:FEEDS]->(d:Tier) "
            "MATCH (a:Company)-[e]->(b:Company) WHERE a.tier = s.tier AND b.tier = d.tier "
            "RETURN s.tier AS src, d.tier AS dst, count(DISTINCT a.id) AS sellers, "
            "count(DISTINCT b.id) AS buyers, count(e) AS edges, "
            "sum(CASE WHEN e.accession_no IS NULL THEN 1 ELSE 0 END) AS uncited"
        ):
            src, dst = rows.get(row["src"]), rows.get(row["dst"])
            if src:
                src["out_edges"] += row["edges"]
                src["out_uncited"] += row["uncited"]
                src["feeds"].append({"tier": row["dst"], "edges": row["edges"],
                                     "buyers": row["buyers"], "uncited": row["uncited"]})
            if dst:
                dst["in_edges"] += row["edges"]
                dst["in_uncited"] += row["uncited"]
                dst["fed_by"].append({"tier": row["src"], "edges": row["edges"],
                                      "sellers": row["sellers"], "uncited": row["uncited"]})

        summaries = []
        for tier in sorted(rows, reverse=True):
            item = rows[tier]
            item["feeds"].sort(key=lambda f: -f["edges"])
            item["fed_by"].sort(key=lambda f: -f["edges"])
            item["evidence_pct"] = round(100.0 * (item["out_edges"] - item["out_uncited"])
                                         / item["out_edges"], 1) if item["out_edges"] else None
            summaries.append(item)
        return summaries

    def tier_roster(self) -> str:
        """
        The tier list as one line per stratum, for prompts that describe the schema.

        Hand-kept copies of this block drift the moment a company is ingested: the text-to-Cypher
        prompt listed 32 tickers long after the graph held 47, and a model told "Tier 1: TSM, MU"
        will answer from that sentence instead of from the data.
        """
        lines = []
        for item in self.get_tier_summaries():
            name = item.get("name", "").split(": ", 1)[-1]
            holders = ", ".join(item.get("tickers") or [])
            lines.append(f"- Tier {item['tier']}: {name} ({holders})")
        return "\n".join(lines)

    def get_topology(self, focus_ticker: str, hops: int = 1,
                     show_components: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Ego network around a company, traversed with Cypher."""
        focus = self.resolve_node_id(focus_ticker.upper().strip())
        if not focus:
            return {}, []

        focus_info = self._node_cache.get(focus, {})
        nodes_dict: Dict[str, Dict[str, Any]] = {
            focus: {
                "id": focus,
                "label": focus_info.get("label", focus),
                "color": "#1E88E5",
                "size": 32,
                "tier": focus_info.get("tier", DEFAULT_TIER),
                "type": "Company",
                "role": "focus",
                "hop": 0,
                "title": f"Focus Company: {focus_info.get('full_name', focus)}",
            }
        }
        edges_list: List[Dict[str, Any]] = []
        seen_edges = set()

        def add_edge(source, target, rel, color, notes=""):
            key = (source, target, rel)
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges_list.append({"source": source, "target": target, "type": rel,
                               "color": color, "notes": notes or ""})

        def add_company(node_id, role, hop, color):
            if node_id in nodes_dict:
                return
            info = self._node_cache.get(node_id, {})
            nodes_dict[node_id] = {
                "id": node_id,
                "label": info.get("label", node_id),
                "color": color,
                "size": max(20, 26 - hop * 3),
                "tier": info.get("tier", DEFAULT_TIER),
                "type": "Company",
                "role": role,
                "hop": hop,
                "title": f"{role.title()} (Hop {hop}): {info.get('full_name', node_id)}",
            }

        # One round trip per hop, not two per company: the frontier travels as an id list. The old
        # loop cost 73 Cypher calls to draw NVDA three hops out, and this runs on every click in the
        # graph view. Both directions share the statement because an anchor can be on either end.
        current_layer = {focus}
        visited = {focus}

        for hop in range(1, max(1, hops) + 1):
            if not current_layer:
                break
            rows = self._rows(
                "MATCH (a:Company)-[r:SUPPLIES_TO]->(b:Company) WHERE a.id IN $ids "
                "RETURN a.id AS src, b.id AS dst, r.notes AS notes, 'out' AS dir "
                "UNION ALL "
                "MATCH (a:Company)-[r:SUPPLIES_TO]->(b:Company) WHERE b.id IN $ids "
                "RETURN a.id AS src, b.id AS dst, r.notes AS notes, 'in' AS dir",
                {"ids": sorted(current_layer)},
            )
            next_layer = set()
            # Customers before suppliers, then by id, so a node reachable from several frontier
            # members gets the same role on every render. Frontier iteration order used to decide it.
            for direction in ("out", "in"):
                ordered = sorted((r for r in rows if r["dir"] == direction),
                                 key=lambda r: (r["src"], r["dst"]))
                for row in ordered:
                    src, dst = row["src"], row["dst"]
                    other = dst if direction == "out" else src
                    if direction == "out":
                        add_company(other, "customer", hop, "#E65100" if hop == 1 else "#FFA726")
                        add_edge(src, dst, "SUPPLIES_TO", "#FB8C00", row.get("notes"))
                    else:
                        add_company(other, "supplier", hop, "#2E7D32" if hop == 1 else "#66BB6A")
                        add_edge(src, dst, "SUPPLIES_TO", "#43A047", row.get("notes"))
                    if other not in visited:
                        visited.add(other)
                        next_layer.add(other)
            current_layer = next_layer

        # Capital / partnership neighbours of the companies visited so far (one extra hop).
        # Both directions matter: an anchor can be the target of a partnership edge (ORCL).
        anchors = list(nodes_dict)
        partner_rows = self._rows(
            "MATCH (a:Company)-[r]->(b:Company) "
            "WHERE a.id IN $ids AND (type(r) = 'INVESTED_IN' OR type(r) = 'PARTNERS_WITH') "
            "RETURN a.id AS source, b.id AS target, type(r) AS rel, r.notes AS notes",
            {"ids": anchors},
        ) + self._rows(
            "MATCH (a:Company)-[r]->(b:Company) "
            "WHERE b.id IN $ids AND (type(r) = 'INVESTED_IN' OR type(r) = 'PARTNERS_WITH') "
            "RETURN a.id AS source, b.id AS target, type(r) AS rel, r.notes AS notes",
            {"ids": anchors},
        )
        for row in partner_rows:
            for endpoint in (row["source"], row["target"]):
                if endpoint not in nodes_dict:
                    add_company(endpoint, "partner", 1,
                                RELATION_EDGE_COLORS.get(row["rel"], "#8E24AA"))
            add_edge(row["source"], row["target"], row["rel"],
                     RELATION_EDGE_COLORS.get(row["rel"], "#8E24AA"), row.get("notes"))

        return nodes_dict, edges_list

    def get_suppliers_and_customers(self, ticker: str) -> Dict[str, List[str]]:
        """Direct suppliers, customers, dependencies and capital links for a company."""
        node_id = self.resolve_node_id(ticker.upper().strip()) or ticker.upper().strip()
        # Five statement kinds, one round trip. This runs once per company mentioned in a retrieved
        # chunk, so the old shape paid five queries for every ticker in the RAG context.
        buckets = {
            "sup": [], "cust": [], "inv": [], "invby": [], "part": [],
        }
        rows = self._rows(
            "MATCH (a:Company)-[:SUPPLIES_TO]->(b:Company {id: $id}) RETURN 'sup' AS bucket, a.id AS id "
            "UNION ALL "
            "MATCH (a:Company {id: $id})-[:SUPPLIES_TO]->(b:Company) RETURN 'cust' AS bucket, b.id AS id "
            "UNION ALL "
            "MATCH (a:Company {id: $id})-[:INVESTED_IN]->(b:Company) RETURN 'inv' AS bucket, b.id AS id "
            "UNION ALL "
            "MATCH (a:Company)-[:INVESTED_IN]->(b:Company {id: $id}) RETURN 'invby' AS bucket, a.id AS id "
            "UNION ALL "
            "MATCH (a:Company {id: $id})-[:PARTNERS_WITH]-(b:Company) RETURN 'part' AS bucket, b.id AS id",
            {"id": node_id},
        )
        for row in rows:
            buckets[row["bucket"]].append(row["id"])
        return {
            "ticker": node_id,
            "suppliers": sorted(set(buckets["sup"])),
            "customers": sorted(set(buckets["cust"])),
            "dependencies": [],
            "investments": sorted(set(buckets["inv"])),
            "invested_by": sorted(set(buckets["invby"])),
            "partners": sorted(set(buckets["part"])),
        }

    # ------------------------------------------------------------------
    # Natural language to Cypher
    # ------------------------------------------------------------------
    def schema_profile(self) -> Dict[str, Any]:
        """
        What the store actually holds, discovered by asking the store.

        Both prompts described the schema by hand and both had drifted: the studio prompt promised a
        `name` property the nodes never carried, and it did not mention `:Filing` at all, so
        "which companies filed a 13G on Nvidia" became an INVESTED_IN query that returned nothing and
        looked like an answer. Four statements, ~10 ms, no cache: an ingest changes what exists and a
        stale profile is the exact bug this prevents.
        """
        labels: Dict[str, Dict[str, Any]] = {}
        for row in self._rows("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n"):
            labels[row["label"]] = {"label": row["label"], "count": row["n"], "properties": set()}
        for row in self._rows(
            "MATCH (n) RETURN DISTINCT labels(n) AS l, keys(n) AS k LIMIT 400"
        ):
            for label in row.get("l") or []:
                if label in labels:
                    labels[label]["properties"].update(row.get("k") or [])

        rels: Dict[tuple, Dict[str, Any]] = {}
        for row in self._rows(
            "MATCH (a)-[r]->(b) RETURN type(r) AS t, labels(a) AS la, labels(b) AS lb, count(r) AS n"
        ):
            rels[row["t"]] = {"type": row["t"], "from": row.get("la") or [],
                              "to": row.get("lb") or [], "count": row["n"], "properties": set()}
        for row in self._rows("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t, keys(r) AS k LIMIT 400"):
            if row["t"] in rels:
                rels[row["t"]]["properties"].update(row.get("k") or [])

        for item in list(labels.values()) + list(rels.values()):
            item["properties"] = sorted(item["properties"] - {"embedding"})
        # The value sets a model cannot guess. Filing.form holds 'SC 13G', so a query for '13G'
        # matches nothing and still looks like an answer; the same goes for the layer names.
        for label, prop, limit in (("Filing", "form", 12), ("Company", "layer", 14)):
            if label not in labels:
                continue
            values = self._rows(
                f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                f"RETURN n.{prop} AS v, count(*) AS n ORDER BY n DESC LIMIT {limit}"
            )
            if values:
                labels[label].setdefault("values", {})[prop] = [
                    (row["v"], row["n"]) for row in values]
        return {"labels": [labels[k] for k in sorted(labels)],
                "relationships": [rels[k] for k in sorted(rels)]}

    def schema_block(self, profile: Optional[Dict[str, Any]] = None) -> str:
        """The profile as prompt lines, so no schema sentence is ever typed twice."""
        profile = profile or self.schema_profile()
        nodes = []
        for item in profile["labels"]:
            line = "- (:%s) %d node(s): %s" % (item["label"], item["count"],
                                              ", ".join(item["properties"]) or "no properties")
            for prop, values in (item.get("values") or {}).items():
                line += "\n    %s.%s holds: %s" % (
                    item["label"], prop, ", ".join(f"{v!r} ({n})" for v, n in values))
            nodes.append(line)
        rels = [
            "- (:%s)-[:%s]->(:%s) %d relationship(s): %s" % (
                "/".join(rel["from"]), rel["type"], "/".join(rel["to"]), rel["count"],
                ", ".join(rel["properties"]) or "no properties")
            for rel in profile["relationships"]
        ]
        return "\n".join(nodes + [""] + rels)

    def property_gaps(self, cypher: str,
                      profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Properties a query reads that the store does not carry.

        Relate answers these with nulls instead of an error, which is the worst possible failure:
        `ORDER BY r.filing_date` on edges that store `as_of` returned nine rows in an arbitrary
        order, and the studio showed a confident table. Empty results and parse errors are loud;
        this is the class that is silent, so it gets checked by name against the live profile.
        """
        profile = profile or self.schema_profile()
        node_props = {item["label"]: set(item["properties"]) for item in profile["labels"]}
        rel_props = {rel["type"]: set(rel["properties"]) for rel in profile["relationships"]}

        bound: Dict[str, str] = {}
        for alias, label in re.findall(r"\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z]\w*)", cypher):
            bound[alias] = label
        for alias, rel in re.findall(r"\[\s*([A-Za-z_]\w*)\s*:\s*([A-Z][A-Z_]*)", cypher):
            bound[alias] = rel

        gaps: List[Dict[str, Any]] = []
        seen = set()
        for alias, prop in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b(?!\s*\()", cypher):
            name = bound.get(alias)
            if not name:
                continue
            known = node_props[name] if name in node_props else rel_props.get(name)
            if known is None or prop in known or (alias, prop) in seen:
                continue
            seen.add((alias, prop))
            gaps.append({"ref": f"{alias}.{prop}", "bound_to": name, "available": sorted(known)})
        return gaps
    def execute_cypher(self, cypher_query: str,
                       profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes read-only Cypher on Relate and builds a visual subgraph."""
        clean_query = cypher_query.strip().rstrip(";")
        violation = read_only_violation(clean_query)
        if violation:
            logger.warning("Refused a %s statement on the read-only Cypher path.", violation)
            return {
                "status": "error",
                "message": (f"Refused: {violation} - this path only reads the graph, so a generated "
                            "query cannot edit the store."),
                "refused": True,
                "records": [],
                "nodes": {},
                "edges": [],
                "count": 0,
            }
        logger.info("Executing Cypher on Relate: %s", clean_query)
        try:
            records = self._rows(clean_query)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "records": [],
                "nodes": {},
                "edges": [],
                "count": 0,
            }

        node_ids = self._collect_node_ids(records)
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        if node_ids:
            node_rows = self._rows(
                "MATCH (n:Company) WHERE n.id IN $ids "
                "RETURN n.id AS id, n.label AS label, n.tier AS tier, n.full_name AS full_name",
                {"ids": list(node_ids)},
            )
            for row in node_rows:
                tier = row["tier"] if row.get("tier") is not None else DEFAULT_TIER
                nodes_dict[row["id"]] = {
                    "id": row["id"],
                    "label": row.get("label") or row["id"],
                    "color": TIER_DEFINITIONS.get(tier, {}).get("color", "#1565C0"),
                    "size": 24,
                    "tier": tier,
                    "type": "Company",
                    "title": f"{row['id']} ({TIER_DEFINITIONS.get(tier, {}).get('name', '')})",
                }
            edge_rows = self._rows(
                "MATCH (a:Company)-[r]->(b:Company) WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, b.id AS target, type(r) AS rel, r.notes AS notes",
                {"ids": list(node_ids)},
            )
            for row in edge_rows:
                edges_list.append({
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["rel"],
                    "color": RELATION_EDGE_COLORS.get(row["rel"], "#00C853"),
                    "notes": row.get("notes") or "",
                })

        return {
            "status": "success",
            "records": records,
            "nodes": nodes_dict,
            "edges": edges_list,
            "count": len(records),
            "empty": not records,
            "property_gaps": self.property_gaps(clean_query, profile or self.schema_profile()),
        }

    def _collect_node_ids(self, records: List[Dict[str, Any]]) -> set:
        """Finds node ids referenced by a query result (strings or returned node maps)."""
        known = set(self._node_cache)
        found: set = set()

        def visit(value):
            if isinstance(value, str):
                if value in known:
                    found.add(value)
            elif isinstance(value, dict):
                if value.get("id") in known:
                    found.add(value["id"])
                for inner in value.values():
                    visit(inner)
            elif isinstance(value, (list, tuple)):
                for inner in value:
                    visit(inner)

        for record in records:
            for value in record.values():
                visit(value)
        return found

    # ------------------------------------------------------------------
    # GraphRAG
    # ------------------------------------------------------------------
    def search_graphrag_context(self, query_text: str, query_embedding: List[float],
                                top_k: int = 3) -> Dict[str, Any]:
        """
        Fused vector + BM25 retrieval over the Chunk nodes in Relate, then each hit's ticker
        is expanded with its graph neighbourhood. Everything comes from the database: there is
        no Python-side scan of an embedding list.

        A company named in the question scopes the search, and a chunk below MIN_VECTOR_SIM is
        dropped rather than passed on: an unrelated excerpt in the prompt is what makes a model
        answer a question it has no evidence for.
        """
        entities = self.find_entities(query_text)
        hits = self.hybrid_search_chunks(query_text, query_embedding, top_k=top_k,
                                         tickers=entities[:2] or None)
        chunks = [
            {
                "id": hit["id"],
                "text": hit["text"],
                "ticker": hit["ticker"],
                "accession_no": hit["accession_no"],
                "form": hit.get("form"),
                "as_of": hit.get("as_of"),
                "heading": hit.get("heading"),
                "rrf": round(float(hit.get("rrf") or 0.0), 5),
                "similarity": hit.get("v_score"),
                "in_vector": hit.get("v_rank"),
                "in_fulltext": hit.get("f_rank"),
                "tiers": [t for t in (hit.get("tiers") or []) if t is not None],
            }
            for hit in hits
        ]
        # One context per ticker, in fused-rank order: the same supplier listed in three
        # excerpts should not repeat the same neighbourhood three times.
        seen: List[str] = []
        for chunk in chunks:
            ticker = chunk.get("ticker")
            if ticker and ticker not in seen:
                seen.append(ticker)
        # Nothing cleared the gate. Fall back to the company the question actually named - the
        # graph neighbourhood is still evidence - instead of the first node in the cache, which
        # used to hand the model an arbitrary company's supply chain to describe.
        if not seen:
            seen = entities[:1]
        return {
            "chunks": chunks,
            "graph_contexts": [self.get_suppliers_and_customers(t) for t in seen],
            "entities": entities,
            "unknown_entities": self.unindexed_entities(query_text),
        }
