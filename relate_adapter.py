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
import json
import logging
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

# Architectural Tier Definitions for the AI Stack View.
# Tier -1 sits *below* equipment: the materials and specialty inputs the whole stack consumes.
TIER_DEFINITIONS = {
    -1: {"name": "Tier -1: Raw Materials & Specialty Inputs", "color": "#37474F", "edge_color": "#607D8B"},
    0: {"name": "Tier 0: Equipment, Power & Facilities", "color": "#5D4037", "edge_color": "#8D6E63"},
    1: {"name": "Tier 1: Foundries, Packaging & Memory", "color": "#2E7D32", "edge_color": "#43A047"},
    2: {"name": "Tier 2: AI Accelerators & Silicon Design", "color": "#1565C0", "edge_color": "#1E88E5"},
    3: {"name": "Tier 3: AI Server Integrators & Systems", "color": "#6A1B9A", "edge_color": "#8E24AA"},
    4: {"name": "Tier 4: Hyperscalers & AI Cloud", "color": "#E65100", "edge_color": "#FB8C00"},
}

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
        self._load_chunks()
        if self.count_nodes() == 0:
            self.seed_ai_supply_chain()
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
        """Maps a free-text entity name onto an existing node id (case-insensitive)."""
        if not name:
            return None
        row = self._rows("MATCH (n:Company {id: $id}) RETURN n.id AS id LIMIT 1", {"id": name})
        if row:
            return row[0]["id"]

        key = self._canonical_key(name)
        for node_id in self._node_cache:
            if self._canonical_key(node_id) == key:
                return node_id
        return None

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
            "n.full_name = $full_name, n.description = $description",
            {
                "id": node_id,
                "label": properties["label"],
                "type": node_type,
                "tier": resolved_tier,
                "tier_name": properties["tier_name"],
                "full_name": props.get("full_name"),
                "description": props.get("description"),
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
        for key in ("notes", "source_form", "accession_no", "as_of", "stake_pct"):
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
            "ON CREATE SET c.label = $ticker, c.type = 'Company', c.tier = $tier, c.tier_name = $tier_name "
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
    # Chunks (vector sidecar) - embeddings stay in JSON; graph context comes from Relate
    # ------------------------------------------------------------------
    def _load_chunks(self):
        """Loads embedded text chunks and vectors for GraphRAG retrieval."""
        self.chunks: List[Dict[str, Any]] = []
        if os.path.exists(self.chunks_file):
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as handle:
                    self.chunks = json.load(handle)
            except Exception as exc:
                logger.warning("Could not load chunks file: %s", exc)

    def save_chunk(self, chunk_id: str, text: str, ticker: str, accession_no: str,
                   embedding: List[float]):
        """Stores a filing excerpt chunk and its embedding vector for GraphRAG retrieval."""
        existing = next((c for c in self.chunks if c["id"] == chunk_id), None)
        if existing:
            existing.update({
                "text": text,
                "embedding": embedding,
                "ticker": ticker,
                "accession_no": accession_no,
            })
        else:
            self.chunks.append({
                "id": chunk_id,
                "text": text,
                "ticker": ticker,
                "accession_no": accession_no,
                "embedding": embedding,
            })
        try:
            with open(self.chunks_file, "w", encoding="utf-8") as handle:
                json.dump(self.chunks, handle, indent=2)
        except Exception as exc:
            logger.warning("Could not save chunks: %s", exc)

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
                "n.tier AS tier, n.tier_name AS tier_name, n.full_name AS full_name, "
                "n.description AS description"
            )
            edges = self._rows(
                "MATCH (a:Company)-[r]->(b:Company) RETURN a.id AS source, b.id AS target, "
                "type(r) AS type, r.notes AS notes, r.source_form AS source_form, "
                "r.accession_no AS accession_no, r.as_of AS as_of, r.stake_pct AS stake_pct"
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

    # Backwards-compatible name used by ingestion.
    def _persist(self):
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
                "n.tier_name = $tier_name, n.full_name = $full_name, n.description = $description",
                {
                    "id": node_id,
                    "label": info.get("label") or node_id,
                    "type": node_type,
                    "tier": tier if tier is not None else infer_tier(node_id, node_type),
                    "tier_name": info.get("tier_name")
                    or TIER_DEFINITIONS.get(tier if tier is not None else DEFAULT_TIER, {}).get("name", ""),
                    "full_name": info.get("full_name"),
                    "description": info.get("description"),
                },
            )
            imported_nodes += 1

        imported_edges = 0
        for edge in edges:
            rel = edge.get("type")
            if rel not in RELATION_TYPES:
                continue
            props = {key: edge.get(key) for key in
                     ("notes", "source_form", "accession_no", "as_of", "stake_pct")
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

        current_layer = {focus}
        visited = {focus}

        for hop in range(1, max(1, hops) + 1):
            next_layer = set()
            for node_id in current_layer:
                downstream = self._rows(
                    "MATCH (a:Company {id: $id})-[r:SUPPLIES_TO]->(b:Company) "
                    "RETURN b.id AS id, r.notes AS notes",
                    {"id": node_id},
                )
                for row in downstream:
                    add_company(row["id"], "customer", hop, "#E65100" if hop == 1 else "#FFA726")
                    add_edge(node_id, row["id"], "SUPPLIES_TO", "#FB8C00", row.get("notes"))
                    if row["id"] not in visited:
                        visited.add(row["id"])
                        next_layer.add(row["id"])

                upstream = self._rows(
                    "MATCH (a:Company)-[r:SUPPLIES_TO]->(b:Company {id: $id}) "
                    "RETURN a.id AS id, r.notes AS notes",
                    {"id": node_id},
                )
                for row in upstream:
                    add_company(row["id"], "supplier", hop, "#2E7D32" if hop == 1 else "#66BB6A")
                    add_edge(row["id"], node_id, "SUPPLIES_TO", "#43A047", row.get("notes"))
                    if row["id"] not in visited:
                        visited.add(row["id"])
                        next_layer.add(row["id"])

            current_layer = next_layer
            if not current_layer:
                break

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
        suppliers = [r["id"] for r in self._rows(
            "MATCH (a:Company)-[:SUPPLIES_TO]->(b:Company {id: $id}) RETURN a.id AS id ORDER BY id",
            {"id": node_id})]
        customers = [r["id"] for r in self._rows(
            "MATCH (a:Company {id: $id})-[:SUPPLIES_TO]->(b:Company) RETURN b.id AS id ORDER BY id",
            {"id": node_id})]
        investments = [r["id"] for r in self._rows(
            "MATCH (a:Company {id: $id})-[:INVESTED_IN]->(b:Company) RETURN b.id AS id ORDER BY id",
            {"id": node_id})]
        invested_by = [r["id"] for r in self._rows(
            "MATCH (a:Company)-[:INVESTED_IN]->(b:Company {id: $id}) RETURN a.id AS id ORDER BY id",
            {"id": node_id})]
        partners = [r["id"] for r in self._rows(
            "MATCH (a:Company {id: $id})-[r:PARTNERS_WITH]-(b:Company) RETURN b.id AS id",
            {"id": node_id})]
        return {
            "ticker": node_id,
            "suppliers": sorted(set(suppliers)),
            "customers": sorted(set(customers)),
            "dependencies": [],
            "investments": sorted(set(investments)),
            "invested_by": sorted(set(invested_by)),
            "partners": sorted(set(partners)),
        }

    # ------------------------------------------------------------------
    # Natural language to Cypher
    # ------------------------------------------------------------------
    def execute_cypher(self, cypher_query: str) -> Dict[str, Any]:
        """Executes arbitrary read-only Cypher on Relate and builds a visual subgraph."""
        clean_query = cypher_query.strip().rstrip(";")
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
        Retrieves top filing chunks by cosine similarity, then expands each ticker with
        its Relate graph neighbourhood. Embeddings live in a JSON sidecar; the graph
        context itself is always queried from Relate.
        """
        if not self.chunks or not query_embedding:
            mentioned = [node_id for node_id in self._node_cache
                         if node_id.lower() in (query_text or "").lower()]
            fallback = mentioned[0] if mentioned else next(iter(self._node_cache), "NVDA")
            return {"chunks": [], "graph_contexts": [self.get_suppliers_and_customers(fallback)]}

        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector) or 1.0

        scored = []
        for chunk in self.chunks:
            chunk_vector = np.array(chunk["embedding"], dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vector) or 1.0
            scored.append((float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm)), chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[:top_k]]
        graph_contexts = [
            self.get_suppliers_and_customers(chunk.get("ticker", "NVDA"))
            for chunk in top_chunks
        ]
        return {"chunks": top_chunks, "graph_contexts": graph_contexts}
