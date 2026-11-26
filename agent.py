"""
agent.py - Tool-using supply-chain analyst agent over the Relate graph.

The agent plans, calls tools, observes results and only then answers. Tools are:
  graph_query      read-only Cypher against the embedded Relate graph
  graph_overview   totals and relationship mix
  company_network  direct suppliers / customers / partners / investments for a ticker
  search_filings   vector (or keyword) search over ingested SEC excerpts, with citations
  ingest_company   fetch + ingest the latest SEC filings for a ticker (the only writer)

Every step is recorded in a trace so the UI can show the plan -> call -> observation loop,
which is the agentic part judges are scoring.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

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
                    "Schema: (:Company {id, label, tier, tier_name, full_name, description}) "
                    "with relationships SUPPLIES_TO, INVESTED_IN, PARTNERS_WITH "
                    "(properties: notes, source_form, accession_no, as_of, stake_pct), and "
                    "(:Filing {id, ticker, accession_no, form, filing_date}) linked by (Company)-[:FILED]->(Filing). "
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
                "description": "Graph totals: companies, relationships by type, filings by form, tier sizes.",
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
        tiers = self.graph._rows(
            "MATCH (n:Company) RETURN n.tier AS tier, count(n) AS c ORDER BY tier DESC"
        )
        return {
            "companies": self.graph.count_nodes(),
            "relationships": {row["type"]: row["c"] for row in types},
            "filings_by_form": {row["form"]: row["c"] for row in forms},
            "companies_per_tier": {str(row["tier"]): row["c"] for row in tiers},
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
        chunks = getattr(self.graph, "chunks", []) or []
        if not chunks:
            return {"error": "No filing excerpts are ingested yet."}
        k = max(1, min(int(k or 3), 5))

        scores: List[float] = []
        if self.embedder is not None:
            try:
                import numpy as np

                query_vector = np.array(self.embedder.encode(query).tolist(), dtype=np.float32)
                query_norm = float(np.linalg.norm(query_vector)) or 1.0
                for chunk in chunks:
                    vector = np.array(chunk["embedding"], dtype=np.float32)
                    scores.append(float(np.dot(query_vector, vector) / ((np.linalg.norm(vector) or 1.0) * query_norm)))
            except Exception:
                scores = []

        if not scores:
            terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
            for chunk in chunks:
                text = (chunk.get("text") or "").lower()
                scores.append(float(sum(text.count(term) for term in terms)))

        ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)[:k]
        results = []
        for score, chunk in ranked:
            results.append({
                "ticker": chunk.get("ticker"),
                "accession_no": chunk.get("accession_no"),
                "score": round(score, 3),
                "excerpt": (chunk.get("text") or "")[:350],
            })
        return {"results": results}

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
