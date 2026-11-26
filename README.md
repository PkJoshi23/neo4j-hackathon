# ⚡ AI Supply Chain Stock Assistant
### Built for the [Relate by Neo4j Hackathon](https://relate.devpost.com/)

A token-efficient AI Supply Chain Stock Assistant powered by **Relate by Neo4j** (embedded graph database), **DeepSeek API**, **SEC EDGAR filings**, and **Streamlit**.

It maps the **"AI Money Cycle"** (Hyperscaler CapEx ➔ AI Accelerators ➔ Foundries, HBM Memory, & Power/Cooling infrastructure) to enable investors and equity analysts to query upstream suppliers, downstream customers, and second-order bottleneck risks with full SEC filing auditability.

**Relate is the source of truth.** Every node, relationship, filing record and tier the UI shows is read from the embedded graph database with Cypher; `supply_chain.json` is only an export/debug snapshot. The app renders the Cypher it executed underneath every chart.

---

## 🧠 How Relate is used

`relate.threadsafety == 1`, so a connection may only be used by the thread that opened it — and Streamlit reruns on a fresh thread each time. The app therefore owns the connection on a single-worker executor (`RelateStore`) and dispatches every statement there, which also gives it a query log for the UI.

| View / feature | Cypher that powers it |
|---|---|
| AI Stack chart | `MATCH (n:Company) WHERE n.tier IN $tiers …` + `MATCH (a:Company)-[r]->(b:Company) WHERE … type(r) IN ['SUPPLIES_TO','INVESTED_IN','PARTNERS_WITH']` |
| Focus Network | per-hop `MATCH (a:Company {id:$id})-[:SUPPLIES_TO]->(b:Company)` in both directions, then `INVESTED_IN` / `PARTNERS_WITH` neighbours |
| Analyst chat context | `MATCH (a:Company {id:$id})-[:SUPPLIES_TO]->(b:Company)` etc. per retrieved ticker |
| Text-to-Cypher studio | the generated query is executed **natively** by Relate, then its matched nodes/edges are re-read from the graph |
| Ingestion idempotency | `MATCH (f:Filing {id:$id})` before any document is re-read |

The **"🔎 Cypher executed against Relate"** panel at the bottom of each chart shows the exact statements, row counts and latency — the receipt that the graph database is doing the work, not a JSON file.

---

## 🏆 Key Features

### 1. 🏢 The "AI Stack" View (Tiered Subgraph Clustering)
Visualizes the AI supply chain mapped into **hierarchical architectural strata** (listed companies only):
- **Tier 4 (Hyperscalers & AI Cloud):** MSFT, AMZN, GOOGL, META, AAPL, ORCL, CRWV, IREN
- **Tier 3 (AI Server Integrators & Systems):** SMCI, DELL, HPE
- **Tier 2 (AI Accelerators & Silicon Design):** NVDA, AVGO, AMD, ANET, MRVL, INTC, ALAB
- **Tier 1 (Foundries, OSAT & Memory):** TSM, MU, SK Hynix, LITE, COHR
- **Tier 0 (Equipment, Power & Facilities):** ASML, AMAT, LRCX, KLAC, VST, CEG, Vertiv
- **Tier -1 (Raw Materials & Specialty Inputs):** CCJ (uranium), TECK (germanium), ENTG (fab materials), LIN (industrial gases), PLAB (photomasks), 4063.T (Shin-Etsu), 3436.T (SUMCO)

Features interactive tier-filtering checkboxes, live node count metrics per tier, and a tier-axis chart (Tier 4 at the top down to Tier 0) where every tier is a labelled row, companies sit on a shared centred axis, and rows are re-ordered to minimise edge crossings. Relationship types are distinguished by line style instead of on-edge text labels, with a legend and hover tooltips. Every chart carries the **Cypher executed against Relate** panel, and the sidebar always shows the live graph size (companies / relationships / filings).

### 2. 🔮 Natural Language to Cypher (Text-to-Graph)
Ask plain-English questions (e.g., *"Who supplies directly to Microsoft and Amazon?"* or *"Which companies invest in Marvell?"*). 
- DeepSeek automatically translates the question into valid **Cypher 25** queries for Relate.
- Displays the generated Cypher query in a code block.
- Executes natively against Relate and renders both the **matched visual sub-graph** and an **interactive data table**.

### 3. 🔍 Focus Network (Ego View)
Select any focus ticker (default: `NVDA`) and dynamically explore 1-to-3 hops of upstream suppliers (Green) and downstream customers (Orange), laid out as a left-to-right supply-chain rail (suppliers → focus → customers), with a **PARTNERS & INVESTORS** row showing capital/partnership links extracted from 8-K, 13D/13G and proxy filings.

### 4. 🤖 Agentic Supply Chain Analyst (tool-calling loop)
The chat tab runs a **plan → tool call → observation → answer** loop instead of a single prompt. The agent decides which evidence to gather, and every step is shown in a **Tool trace** expander with arguments, result summary, latency and token count.

| Tool | What it does |
|---|---|
| `graph_query(cypher)` | Runs **read-only** Cypher against Relate (writes are rejected), returns rows plus the matched subgraph |
| `graph_overview()` | Graph totals: companies, relationship mix, filings by form, tier sizes |
| `company_network(ticker)` | Direct suppliers, customers, partners, investments and investors |
| `search_filings(query, k)` | Retrieval over ingested SEC excerpts, returning ticker + accession number for citations |
| `ingest_company(ticker)` | Fetches and ingests the latest SEC filings for a ticker (the only write tool) |

Answers cite the filing form, accession number and date, distinguish supply edges from capital/partnership edges, and state plainly when the graph or filings do not contain the answer. The previous one-shot GraphRAG analyst remains available as a fast mode in the same tab.

### 5. 📥 Token-Efficient Live SEC Ingestion
Zero-token idempotency gate + local regex pre-filtering sieve for SEC filings via `edgartools`.

**Form coverage** — supply chain and capital facts live in different documents, so each ingest reads:
- **10-K / 20-F** — business & risk sections (wafer, foundry, customer concentration)
- **8-K** — material events, including the **EX-99.x press-release exhibits** where partnerships and investments are announced
- **SC 13D / SC 13G** — equity stakes, read from structured metadata (issuer + reporting person CIK): **zero tokens**
- **DEF 14A** — related-party transactions from the proxy statement

**Cost and latency controls**
- per-form character caps (12k annual, 6k 8-K, 6k proxy) and filing limits, with a 28k character budget per run
- a local regex sieve (supply-chain + ownership keywords) — a filing with no signal costs **0 tokens**
- one LLM call per evidence document, run concurrently, so a full company refresh takes seconds
- re-running an already ingested company is a metadata-only no-op (≈1 s, 0 tokens)

**Graph hygiene**
- **Listed companies only** — the graph has no component/product/technology nodes; an extracted name must be a real ticker (`NVDA`, `000660.KS`) or a known alias
- **Capital and partnership links** (`INVESTED_IN`, `PARTNERS_WITH`) only connect companies already tracked, so one press release cannot drag in a fan of new nodes
- every extracted edge records its provenance (`source_form`, `accession_no`, `as_of`) and is drawn with its own line style

---

## 🚀 Quick Start for Hackathon Judges

### 1. Clone & Install
```bash
git clone https://github.com/your-username/ai-supply-chain-assistant.git
cd ai-supply-chain-assistant

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate # Linux/macOS

# Install Relate (embedded graph DB) + the rest of the stack
pip install -r requirements.txt

# If relate-db is not on your package index, install it with the official script:
#   Windows:      irm https://relatedb.ai/install.ps1 | iex
#   macOS/Linux:  curl -sSfL https://relatedb.ai/install.sh | bash
```

The repo ships a ready-to-open graph (`supply_chain.rl`, 21 companies / 43 relationships / 106 SEC filings), so the app starts with data on first run.

### 2. Configure Environment (.env)
```ini
# DeepSeek LLM Credentials (OpenAI-compatible)
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# SEC EDGAR Identity (Required by SEC fair access policy)
EDGAR_IDENTITY=SupplyChainBot admin@stockassistant.ai
```
*(No database credentials or server setup required! Relate runs locally in-process).*

### 3. Launch App
```bash
streamlit run app.py
```

---

## 📋 Devpost Submission Tech Stack
- **Database:** Relate by Neo4j (`relate-db`, embedded, single-file graph — the source of truth)
- **Agent:** DeepSeek `deepseek-chat` function calling (5 tools, plan → call → observe loop)
- **LLM:** DeepSeek (`deepseek-chat`) for text-to-Cypher and the GraphRAG analyst
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5` ($0 local API spend)
- **Data Source:** SEC EDGAR via `edgartools` (10-K, 20-F, 8-K + exhibits, 13D/13G, DEF 14A)
- **Visualization:** `streamlit-agraph` + `streamlit`
