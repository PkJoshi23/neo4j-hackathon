"""
agent.py - Tool-using supply-chain analyst agent over the Relate graph.

The agent plans, calls tools, observes results and only then answers. Tools are:
  graph_query      read-only Cypher against the embedded Relate graph
  graph_overview   totals and relationship mix
  layer_concentration per-industrial-layer member count, buyer load and evidence coverage
  company_network  direct suppliers / customers / partners / investments for a ticker
  search_filings   vector (or keyword) search over ingested SEC excerpts, with citations
  ingest_company   fetch + ingest the latest SEC filings for a ticker (the only writer)

Every step is recorded in a trace so the UI can show the plan -> call -> observation loop,
which is the agentic part judges are scoring.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from relate_adapter import embed_query

logger = logging.getLogger(__name__)

READ_ONLY_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL|ALTER|GRANT)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a Senior AI Supply Chain Equity Research Analyst agent.

You answer questions about the AI supply chain (hyperscalers, accelerators, foundries,
memory, equipment) from TWO evidence sources:
1. the embedded Relate graph (companies and their SUPPLIES_TO / INVESTED_IN / PARTNERS_WITH
   relationships, each with provenance: source form, accession number, date);
2. ingested SEC filing excerpts (10-K, 20-F, 8-K including press-release exhibits, 13D/13G,
   DEF 14A).

How to work:
- Plan first, then call tools. Prefer graph tools for "who connects to whom", and
  search_filings for "what evidence supports this".
- Be economical: prefer ONE well-targeted query over exploratory ones, and stop at three
  tool calls unless the evidence you need is genuinely missing.
- Never invent a company, relationship, number or quote. If the tools do not show it, say
  plainly that it is not in the graph or the ingested filings.
- Cite evidence with the filing form + accession number + date exactly as the tools return it.
- Relationships are DIRECTED. Never claim an edge exists in both directions unless the tool
  returns two rows (one per direction). A partner query that returns one row means one edge.
- Distinguish supply-chain edges from capital/partnership edges (INVESTED_IN, PARTNERS_WITH).
- For "which layer is the chokepoint", call layer_concentration: it reports each industrial
  layer's member count, which supplier carries the most buyer relationships, and how much of
  the layer is backed by a filing.
- For a question about a stratum ("who feeds the hyperscalers", "is tier 3 thin"), call
  graph_overview rather than counting companies yourself: tier membership and the flows between
  tiers are already in the graph as (Company)-[:IN_TIER]->(Tier)-[:FEEDS]->(Tier).
- An edge with no accession number was curated by hand, not read out of a filing. Say so when
  you rely on one rather than implying the SEC disclosed it.
- Be concise: a few short paragraphs or bullets, then a short "Evidence" list.
- You may call at most 5 tools. Stop calling tools once you can answer.
"""


def _tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "graph_query",
                "description": (
                    "Run a READ-ONLY Cypher query against the embedded Relate graph. "
                    "Schema: (:Company {id, label, tier, tier_name, full_name, description, layer}) "
                    "with relationships SUPPLIES_TO, INVESTED_IN, PARTNERS_WITH "
                    "(properties: notes, source_form, accession_no, as_of, stake_pct), and "
                    "(:Filing {id, ticker, accession_no, form, filing_date}) linked by (Company)-[:FILED]->(Filing). "
                    "The strata are nodes too: (Company)-[:IN_TIER]->(Tier {tier, name, short, color}) "
                    "and (Tier)-[:FEEDS]->(Tier), so a tier can be matched directly instead of "
                    "re-derived from n.tier. "
                    "Use MATCH / WHERE / RETURN / ORDER BY / LIMIT only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"cypher": {"type": "string", "description": "Cypher statement"}},
                    "required": ["cypher"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_overview",
                "description": (
                    "Graph totals: companies, relationships by type, filings by form, and one row "
                    "per architectural tier with its members, the tiers it feeds and is fed by, and "
                    "what share of its outgoing edges cite a filing."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "layer_concentration",
                "description": (
                    "Per industrial layer (Materials, EDA, Fab Equipment, Foundry, Memory, "
                    "Photonics, Silicon, Test, Systems, Power, Cloud): member count, how many "
                    "buyers its suppliers serve, which company carries the largest share of that "
                    "load, and what percentage of its edges cite a filing. Use this to answer "
                    "which layer is a chokepoint or single point of failure."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "company_network",
                "description": (
                    "Direct network of one company: tier, suppliers, customers, partners, "
                    "investments (it invests in) and investors (invest in it)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"}},
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_filings",
                "description": (
                    "Search ingested SEC filing excerpts for evidence. Returns excerpts with "
                    "ticker, form, accession number and date so they can be cited."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for"},
                        "k": {"type": "integer", "description": "How many excerpts (default 4, max 6)"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ingest_company",
                "description": (
                    "Fetch and ingest the latest SEC filings for a ticker into the graph. "
                    "This WRITES to the graph and costs LLM tokens. Use only when the company "
                    "is missing from the graph or the user explicitly asks to refresh it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string", "description": "Stock ticker to ingest"}},
                    "required": ["ticker"],
                },
            },
        },
    ]


class SupplyChainAgent:
    """A small ReAct-style loop with explicit tool calls and an observable trace."""

    def __init__(self, client, graph, embedder=None, llm_wrapper=None, model: str = "deepseek-chat"):
        self.client = client
        self.graph = graph
        self.embedder = embedder
        self.llm_wrapper = llm_wrapper
        self.model = model

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def _tool_graph_query(self, cypher: str) -> Dict[str, Any]:
        if not cypher or not cypher.strip():
            return {"error": "empty Cypher"}
        if READ_ONLY_FORBIDDEN.search(cypher):
            return {"error": "Rejected: graph_query is read-only (no CREATE/MERGE/SET/DELETE)."}
        result = self.graph.execute_cypher(cypher)
        if result.get("status") == "error":
            return {"error": result["message"]}
        records = result["records"]
        return {
            "rows": len(records),
            "records": records[:10],
            "truncated": len(records) > 10,
            "matched_nodes": len(result["nodes"]),
            "matched_edges": len(result["edges"]),
        }

    def _tool_graph_overview(self) -> Dict[str, Any]:
        types = self.graph._rows(
            "MATCH (a:Company)-[r]->(b:Company) RETURN type(r) AS type, count(*) AS c ORDER BY c DESC"
        )
        forms = self.graph._rows(
            "MATCH (f:Filing) RETURN f.form AS form, count(*) AS c ORDER BY c DESC"
        )
        tiers = self.graph.get_tier_summaries()
        return {
            "companies": self.graph.count_nodes(),
            "relationships": {row["type"]: row["c"] for row in types},
            "filings_by_form": {row["form"]: row["c"] for row in forms},
            "tiers": [{
                "tier": tier["tier"],
                "name": tier["name"],
                "members": tier.get("members", 0),
                "tickers": tier.get("tickers") or [],
                "feeds": tier["feeds"],
                "fed_by": tier["fed_by"],
                "outgoing_edges": tier["out_edges"],
                "uncited_outgoing": tier["out_uncited"],
                "evidence_pct": tier["evidence_pct"],
            } for tier in tiers],
            "read": ("feeds lists the tiers this stratum supplies, with the edge count and how many "
                     "of those edges carry no accession number. evidence_pct covers outgoing edges "
                     "only, because it is the supplier's own filing that names the relationship."),
        }

    def _tool_layer_concentration(self) -> Dict[str, Any]:
        layers = self.graph.get_layer_map()
        if not layers:
            return {"layers": [], "note": "No company carries a layer tag yet."}
        return {
            "layers": layers,
            "read": ("load_share is the fraction of that layer's buyer relationships carried by "
                     "its single largest supplier; evidence_pct is how much of the layer is "
                     "backed by a filing rather than curated by hand."),
        }

    def _tool_company_network(self, ticker: str) -> Dict[str, Any]:
        node_id = self.graph.resolve_node_id((ticker or "").upper().strip())
        if not node_id:
            return {"error": f"{ticker} is not in the graph. Ingest it first or check the ticker."}
        info = self.graph.nodes.get(node_id, {})
        network = self.graph.get_suppliers_and_customers(node_id)
        return {
            "ticker": node_id,
            "tier": info.get("tier"),
            "full_name": info.get("full_name"),
            "suppliers": network["suppliers"],
            "customers": network["customers"],
            "partners": network["partners"],
            "invests_in": network["investments"],
            "invested_by": network["invested_by"],
        }

    def _tool_search_filings(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Delegates to Relate: one fused vector + BM25 statement, not a scan of a Python list.
        """
        query_embedding = None
        if self.embedder is not None:
            try:
                query_embedding = embed_query(self.embedder, query)
            except Exception as exc:
                logger.warning("Could not embed the search query: %s", exc)
        entities = self.graph.find_entities(query)
        hits = self.graph.hybrid_search_chunks(query, query_embedding, top_k=k,
                                               tickers=entities[:2] or None)
        unknown = self.graph.unindexed_entities(query)
        if not hits:
            # Saying "nothing matched" is the useful answer. Without it the model fills the gap
            # with whatever company it already has in context.
            note = "No filing excerpt matches this query."
            if unknown:
                note += (f" {', '.join(unknown)} is not in the graph - no filing has been ingested "
                         "for it, so answer from what is stored or say you do not know.")
            return {"results": [], "note": note, "companies_named": entities,
                    "not_in_graph": unknown}
        return {"results": [
            {
                "ticker": hit.get("ticker"),
                "accession_no": hit.get("accession_no"),
                "form": hit.get("form"),
                "filed": hit.get("as_of"),
                "heading": hit.get("heading"),
                "rrf": round(float(hit.get("rrf") or 0.0), 5),
                "similarity": hit.get("v_score"),
                "found_by": "entity-scope+vector" if hit.get("ticker") in entities else (
                    "vector+fulltext" if hit.get("lists", 0) > 1 else (
                        "fulltext" if hit.get("v_rank") is None else "vector")),
                "excerpt": (hit.get("text") or "")[:350],
            }
            for hit in hits
        ], "companies_named": entities, "not_in_graph": unknown}

    def _tool_ingest_company(self, ticker: str) -> Dict[str, Any]:
        symbol = (ticker or "").upper().strip()
        if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,3})?", symbol):
            return {"error": f"{ticker!r} does not look like a ticker."}
        try:
            import asyncio

            from ingest import efficient_ingest_ticker

            result = asyncio.run(
                efficient_ingest_ticker(symbol, self.graph, self.llm_wrapper, self.embedder)
            )
            return {
                "status": result.get("status"),
                "message": result.get("message"),
                "tokens": result.get("tokens"),
                "forms": result.get("forms"),
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "graph_query":
            return self._tool_graph_query(arguments.get("cypher", ""))
        if name == "graph_overview":
            return self._tool_graph_overview()
        if name == "layer_concentration":
            return self._tool_layer_concentration()
        if name == "company_network":
            return self._tool_company_network(arguments.get("ticker", ""))
        if name == "search_filings":
            return self._tool_search_filings(arguments.get("query", ""), arguments.get("k", 4))
        if name == "ingest_company":
            return self._tool_ingest_company(arguments.get("ticker", ""))
        return {"error": f"Unknown tool {name!r}"}

    @staticmethod
    def _summarise(name: str, result: Dict[str, Any]) -> str:
        if result.get("error"):
            return f"error: {str(result['error'])[:140]}"
        if name == "graph_query":
            return f"{result.get('rows', 0)} row(s), {result.get('matched_nodes', 0)} node(s), {result.get('matched_edges', 0)} edge(s)"
        if name == "graph_overview":
            return f"{result.get('companies', 0)} companies, {sum((result.get('relationships') or {}).values())} relationships"
        if name == "layer_concentration":
            rows = result.get("layers") or []
            top = rows[0] if rows else None
            return (f"{len(rows)} layer(s), most concentrated: "
                    f"{top['layer']} ({top['top_supplier']} = {int(top['load_share'] * 100)}%)"
                    if top else "no layer data")
        if name == "company_network":
            return (f"{result.get('ticker')}: {len(result.get('suppliers', []))} suppliers, "
                    f"{len(result.get('customers', []))} customers, {len(result.get('partners', []))} partners")
        if name == "search_filings":
            tickers = {row.get("ticker") for row in result.get("results", [])}
            return f"{len(result.get('results', []))} excerpt(s) from {', '.join(sorted(t for t in tickers if t)) or 'n/a'}"
        if name == "ingest_company":
            return f"{result.get('status')}: {str(result.get('message'))[:120]}"
        return "ok"

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def run(self, question: str, max_steps: int = 5, token_budget: int = 18_000) -> Dict[str, Any]:
        if self.client is None:
            return {
                "answer": "DeepSeek API key is not configured, so the agent cannot run.",
                "trace": [],
                "citations": [],
                "tokens": 0,
            }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        tokens = 0
        answer = ""

        for step in range(1, max_steps + 1):
            force_final = step == max_steps or tokens >= token_budget
            request = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 900,
            }
            if not force_final:
                request["tools"] = _tool_specs()
                request["tool_choice"] = "auto"
            else:
                messages.append({
                    "role": "user",
                    "content": "Tool budget exhausted. Answer now with what you have, and say what is missing.",
                })

            try:
                response = self.client.chat.completions.create(**request)
            except Exception as exc:
                trace.append({"step": step, "tool": "llm", "args": {}, "summary": f"error: {exc}",
                              "ms": 0, "error": True, "tokens": 0})
                answer = f"The model call failed: {exc}"
                break

            step_tokens = 0
            if getattr(response, "usage", None):
                step_tokens = getattr(response.usage, "total_tokens", 0) or 0
                tokens += step_tokens

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                answer = message.content or ""
                break

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in tool_calls
                ],
            })

            for index, call in enumerate(tool_calls):
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except Exception:
                    arguments = {}

                started = time.perf_counter()
                result = self._execute_tool(name, arguments)
                elapsed = round((time.perf_counter() - started) * 1000, 1)

                for row in (result.get("results") or []):
                    if row.get("accession_no"):
                        citations.append({
                            "ticker": row.get("ticker"),
                            "accession_no": row.get("accession_no"),
                            "form": row.get("form"),
                            "filed": row.get("filed"),
                            "heading": row.get("heading"),
                            "excerpt": (row.get("excerpt") or "")[:200],
                        })

                trace.append({
                    "step": step,
                    "tool": name,
                    "args": arguments,
                    "summary": self._summarise(name, result),
                    "ms": elapsed,
                    "error": bool(result.get("error")),
                    # tokens belong to the model call that requested these tools
                    "tokens": step_tokens if index == 0 else 0,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str)[:3500],
                })

        if not answer:
            answer = "I could not complete the analysis within the tool budget."

        return {
            "answer": answer,
            "trace": trace,
            "citations": citations[:8],
            "tokens": tokens,
        }
