"""
ingest.py - Token-Efficient Multi-Form SEC Ingestion & Entity Resolution for Relate.

Reads several SEC form types so ownership and partnership evidence is not missed:
    10-K / 20-F  business + risk sections        (supply chain prose)
    8-K          material events / press releases (partnerships, agreements)
    SC 13D/13G   equity stakes                    (who invested in whom)
    DEF 14A      related-party transactions       (proxy statement)

Cost and latency control:
- per-form character caps and filing limits
- a local regex sieve drops paragraphs with no supply-chain/ownership signal; a filing
  with no signal costs ZERO LLM tokens
- 13D/13G stakes are read from structured metadata (issuer + reporting person CIK),
  so ownership edges cost zero tokens and no download of the full document
- one LLM call per evidence document, run concurrently, under a per-run char budget
"""

import os
import sys
import types
import re
import json
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# --- Critical Fix for NumPy 2.x vs OpenCV (cv2) compatibility ---
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

# --- Critical Windows Fix for edgartools ---
# edgartools defaults to ~/.edgar which throws PermissionError [WinError 5] on Windows.
LOCAL_EDGAR_DIR = (Path(__file__).parent / ".edgar_data").resolve()
LOCAL_EDGAR_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(LOCAL_EDGAR_DIR))
os.environ.setdefault("EDGAR_CACHE_DIR", str(LOCAL_EDGAR_DIR))

# SEC EDGAR tools import
try:
    from edgar import Company as EdgarCompany, set_identity
    HAS_EDGAR = True
except Exception as e:
    HAS_EDGAR = False
    EdgarCompany = None
    set_identity = None

from relate_adapter import RelateSupplyChainGraph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# SEC EDGAR Identity
DEFAULT_EDGAR_IDENTITY = "SupplyChainBot admin@stockassistant.ai"

# --- Token budget -----------------------------------------------------------
MAX_FILTERED_CHARS = 12_000   # per document
MAX_TOTAL_CHARS = 28_000      # per ingestion run (all documents)
MIN_SIGNAL_CHARS = 400        # below this a document is not worth an LLM call
CHUNKS_PER_FILING = 8         # excerpts stored per filing for GraphRAG
CHUNK_MIN_CHARS = 200         # shorter than this is a caption, not evidence
CHUNK_MAX_CHARS = 1400        # ~350 tokens: past this bge-small dilutes the signal
# Foreign filings (20-F/6-K) arrive hard-wrapped: TSM's 20-F split into 1091 blocks with a
# median length of 115 characters, so the paragraph floor matched three of them and the graph's
# only foundry node got a single excerpt. A document where this share of blocks is short is
# treated as wrapped text and its lines are reflowed before paragraphing.
WRAPPED_DOC_SHARE = 0.75
WRAPPED_BLOCK_CHARS = 180
REFLOW_PARAGRAPH_CHARS = 500  # flush a reflowed paragraph once a sentence ends past this
LLM_CONCURRENCY = 3           # parallel extraction calls

# Canonical ticker mapping for entity resolution
KNOWN_ALIASES: Dict[str, str] = {
    "taiwan semiconductor": "TSM",
    "taiwan semiconductor manufacturing": "TSM",
    "taiwan semiconductor manufacturing company": "TSM",
    "tsmc": "TSM",
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "broadcom": "AVGO",
    "broadcom inc": "AVGO",
    "micron": "MU",
    "micron technology": "MU",
    "micron technology inc": "MU",
    "arista": "ANET",
    "arista networks": "ANET",
    "vertiv": "VRT",
    "vertiv holdings": "VRT",
    "supermicro": "SMCI",
    "super micro": "SMCI",
    "super micro computer": "SMCI",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "asml": "ASML",
    "asml holding": "ASML",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "amazon web services": "AMZN",
    "aws": "AMZN",
    "google cloud": "GOOGL",
    "gcp": "GOOGL",
    "microsoft azure": "MSFT",
    "azure": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "goog": "GOOGL",
    "google inc": "GOOGL",
    "meta platforms": "META",
    "meta": "META",
    "apple": "AAPL",
    "apple inc": "AAPL",
    "intel": "INTC",
    "intel corporation": "INTC",
    "arm holdings": "ARM",
    "arm": "ARM",
    "qualcomm": "QCOM",
    "samsung": "005930.KS",
    "samsung electronics": "005930.KS",
    "sk hynix": "000660.KS",
    "oracle": "ORCL",
    "oracle corp": "ORCL",
    "oracle corporation": "ORCL",
    "coreweave": "CRWV",
    "coreweave inc": "CRWV",
    "marvell": "MRVL",
    "marvell technology": "MRVL",
    "marvell technology inc": "MRVL",
    "lumentum": "LITE",
    "lumentum holdings": "LITE",
    "iris energy": "IREN",
    "iren limited": "IREN",
    "dell": "DELL",
    "dell technologies": "DELL",
    "hewlett packard enterprise": "HPE",
    "hpe": "HPE",
    "coherent": "COHR",
    "coherent corp": "COHR",
    "coherent corporation": "COHR",
    "astera labs": "ALAB",
    "astera": "ALAB",
    "credo": "CRDO",
    "credo technology group": "CRDO",
    "nebius": "NBIS",
    "nebius group": "NBIS",
}

# Signals that a paragraph is about the supply chain.
SUPPLY_CHAIN_KEYWORDS = [
    r"\bsupplier[s]?\b",
    r"\bcustomer[s]?\b",
    r"\bfoundr(?:y|ies)\b",
    r"\bwafer[s]?\b",
    r"\bpackaging\b",
    r"\bsole\s+source\b",
    r"\bsingle\s+source\b",
    r"\bconcentration\b",
    r"\b10%\s+or\s+more\b",
    r"\bhyperscaler[s]?\b",
    r"\bhbm(?:2e|3|3e|4)?\b",
    r"\bcowos\b",
    r"\badvanced\s+packaging\b",
    r"\bcontract\s+manufacturer[s]?\b",
    r"\bdata\s+center\s+infrastructure\b",
    r"\bliquid\s+cooling\b",
    r"\bthermal\s+management\b",
    r"\b(?:tsm|tsmc|nvda|asml|msft|amzn|goog|googl|meta|avgo|amd|mu|smci|vrt|anet|intc)\b",
]

# Signals that a paragraph is about ownership / partnership (where investments live).
OWNERSHIP_KEYWORDS = [
    r"\binvest(?:s|ed|ing|ment|ments|or)\b",
    r"\bequity\s+(?:stake|interest|investment|position)\b",
    r"\b(?:minority|majority|strategic)\s+(?:stake|interest|investment)\b",
    r"\bstake\s+in\b",
    r"\bowns?\b",
    r"\bbeneficial(?:ly)?\s+own(?:s|ed|ership)?\b",
    r"\bjoint\s+venture\b",
    r"\bstrategic\s+(?:partnership|relationship|alliance|collaboration)\b",
    r"\bpartnership\b",
    r"\bmemorandum\s+of\s+understanding\b",
    r"\bcapital\s+commitment\b",
    r"\bpurchase\s+commitment\b",
    r"\brelated[- ]party\s+transaction[s]?\b",
    r"\bshares?\s+of\s+(?:common|preferred)\s+stock\b",
]

SUPPLY_SIEVE = re.compile("|".join(SUPPLY_CHAIN_KEYWORDS), re.IGNORECASE)
OWNERSHIP_SIEVE = re.compile("|".join(OWNERSHIP_KEYWORDS), re.IGNORECASE)
ANY_SIEVE = re.compile("|".join(SUPPLY_CHAIN_KEYWORDS + OWNERSHIP_KEYWORDS), re.IGNORECASE)

# Kept compiled and separate so a paragraph scores once per keyword GROUP rather than once
# per occurrence; otherwise a risk factor that says "customers" fourteen times outscores one
# that actually names a supplier.
_SUPPLY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUPPLY_CHAIN_KEYWORDS]
_OWNERSHIP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in OWNERSHIP_KEYWORDS]
_ALIAS_LIST = sorted({alias for alias, ticker in KNOWN_ALIASES.items() if len(alias) > 3})
_QUANTIFIED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?%|\$\s?\d")
# '--- Item 1A ---' markers written by extract_filing_sections.
_SECTION_MARK = re.compile(r"\n?\s*-{3,}\s*(.+?)\s*-{3,}\s*(?:\n|$)")
# Split only where a terminator is followed by whitespace and a capital/digit/quote, so
# "10% of revenue." and "U.S. sales" do not break a sentence in half.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9$])")
_INLINE_HEAD = re.compile(r"^(?:Item|Part)\s+(?:[0-9]+[A-Za-z]?|[IVX]+)[.:]?\s*(?:\S.*)?$", re.IGNORECASE)
# End of a sentence, allowing a trailing closing quote or bracket.
_TERM_SENT = re.compile(r"[.!?][\"')\]]*\s*$")
# Safe-harbor recitals and 8-K exhibit indexes name counterparties and keywords without claiming
# anything about them ("Certain statements in this press release... supplier relationships"), so
# they win the keyword sieve and rank ahead of the disclosure that actually matters.
_RECITAL = re.compile(
    r"certain statements in this|includes? (?:but is not limited to|statements as to)|"
    r"forward[- ]looking statements? (?:regarding|are based|contained)|"
    r"actual results may (?:differ|vary)|should not be construed|"
    r"expressly qualified by|cautionary statements|"
    r"principal (?:executive )?office|telephone number|facsimile|"
    r"title of each class|trading symbol|name of each exchange|par value per share|"
    r"^\s*(?:\(d\)\s*)?exhibits?\b|item 9\.01\b|incorporated by reference|"
    r"pursuant to the requirements of the securit|intentionally left blank|"
    r"signature[s]?\b",
    re.IGNORECASE | re.MULTILINE,
)


class FormProfile(NamedTuple):
    """How much and how deep to read one form type."""
    kind: str            # annual | event | stake | proxy
    char_cap: int
    filing_limit: int
    items: Optional[Tuple[str, ...]] = None


FORM_PROFILES: Dict[str, FormProfile] = {
    "10-K":    FormProfile("annual", MAX_FILTERED_CHARS, 1, ("Item 1", "Item 1A")),
    "20-F":    FormProfile("annual", MAX_FILTERED_CHARS, 1, ("Item 3", "Item 4")),
    "8-K":     FormProfile("event", 6_000, 2),      # short: material events / press releases
    "SC 13D":  FormProfile("stake", 0, 1),          # structured: zero LLM tokens
    "SC 13G":  FormProfile("stake", 0, 1),
    "DEF 14A": FormProfile("proxy", 6_000, 1),      # huge document, tight cap
}


def init_sec_identity():
    """Initializes the User-Agent identity required by the SEC EDGAR API."""
    identity = os.getenv("EDGAR_IDENTITY", DEFAULT_EDGAR_IDENTITY)
    if set_identity is not None:
        try:
            set_identity(identity)
            logger.info("SEC EDGAR identity initialized as: %s", identity)
        except Exception as e:
            logger.warning("Error setting SEC identity: %s", e)


def extract_filing_sections(filing_obj, form: str) -> str:
    """
    Extracts Business & Risk items (10-K: Items 1/1A; 20-F: Items 3/4).
    Falls back gracefully to plain document text if section parsing fails.
    """
    extracted_text = ""
    target_items = ["Item 1", "Item 1A"] if "10-K" in form.upper() else ["Item 3", "Item 4"]

    try:
        for item in target_items:
            try:
                content = filing_obj[item]
                if content:
                    extracted_text += f"\n--- {item} ---\n{str(content)}"
            except Exception:
                continue
    except Exception as e:
        logger.warning("Item extraction warning: %s", e)

    if len(extracted_text.strip()) < 500:
        try:
            extracted_text = filing_obj.text() if hasattr(filing_obj, "text") else str(filing_obj)
        except Exception:
            extracted_text = str(filing_obj)

    return extracted_text


def _clip_to_sentence(text: str, limit: int) -> str:
    """
    Trim to whole sentences. A hard ``text[:n]`` cut can turn "we do not depend on a single
    supplier" into a stored excerpt that reads as the opposite, and the agent quotes excerpts
    verbatim.
    """
    if len(text) <= limit:
        return text
    kept: List[str] = []
    used = 0
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if kept and used + len(sentence) + 1 > limit:
            break
        if not kept and len(sentence) > limit:
            # One sentence longer than the whole budget: cut at the last clause boundary.
            head = sentence[:limit]
            cut = max(head.rfind("; "), head.rfind(", "), head.rfind(" "))
            kept.append(head[:cut + 1].rstrip(" ,;"))
            break
        kept.append(sentence)
        used += len(sentence) + 1
        if used >= limit:
            break
    return " ".join(kept).strip() or text[:limit].rsplit(" ", 1)[0]


def _fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode("utf-8")).hexdigest()[:16]


def _signal_score(text: str) -> int:
    """Distinct keyword groups + named counterparties + quantified claims, minus boilerplate."""
    supply = sum(1 for pat in _SUPPLY_PATTERNS if pat.search(text))
    ownership = sum(1 for pat in _OWNERSHIP_PATTERNS if pat.search(text))
    named = sum(1 for alias in _ALIAS_LIST if alias in text.lower())
    quantified = len(_QUANTIFIED.findall(text))
    # Measured on 225 ingested chunks: recitals scored 6-8, real disclosure 17-32, so one
    # penalty clears the former without touching the latter.
    recital = min(len(_RECITAL.findall(text)), 2)
    return supply * 3 + ownership * 3 + min(named, 4) * 2 + min(quantified, 4) - recital * 12


def _body_paragraphs(heading: str, body: str):
    """
    Yield (heading, paragraph); a short leading 'Item 1A. Risk Factors' line re-declares it.

    Some filings separate every wrapped line with a blank line instead of every paragraph. When
    most blocks in a section are one line long the lines are reflowed into paragraph-sized runs,
    otherwise the whole document collapses into captions and nothing is retrievable.
    """
    blocks = []
    for block in re.split(r"\n\s*\n", body):
        cleaned = re.sub(r"\s+", " ", block).strip()
        if cleaned:
            blocks.append(cleaned)
    if not blocks:
        return
    wrapped = (sum(1 for b in blocks if len(b) < WRAPPED_BLOCK_CHARS)
               >= WRAPPED_DOC_SHARE * len(blocks))

    current = heading
    para = ""
    for cleaned in blocks:
        if len(cleaned) <= 130 and _INLINE_HEAD.match(cleaned):
            if para:
                yield current, para
                para = ""
            current = cleaned.rstrip(": .-").strip()
            continue
        para = f"{para} {cleaned}" if para else cleaned
        if not wrapped:
            yield current, para
            para = ""
            continue
        if len(para) >= CHUNK_MAX_CHARS or (len(para) >= REFLOW_PARAGRAPH_CHARS
                                            and _TERM_SENT.search(para)):
            yield current, para
            para = ""
    if para:
        yield current, para


def _iter_paragraphs(raw_text: str):
    """
    Walk the document as (heading, paragraph). extract_filing_sections brackets each Item with
    '--- Item 1A ---'; inheriting that label down the document is what lets an excerpt say
    which section it came from instead of floating free.
    """
    heading = ""
    pos = 0
    for mark in _SECTION_MARK.finditer(raw_text):
        yield from _body_paragraphs(heading, raw_text[pos:mark.start()])
        heading = mark.group(1).strip().rstrip(": .-").strip()
        pos = mark.end()
    yield from _body_paragraphs(heading, raw_text[pos:])


def _keyword_windows(raw_text: str, char_cap: int) -> List[Dict[str, Any]]:
    """
    Fallback for flattened documents (8-K HTML often arrives as one long line, so blank-line
    splitting yields nothing). Windows are sentence-bounded and de-overlapped, instead of a
    fixed +/-400 character slice that cuts mid-clause and repeats itself.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    used = 0
    for match in list(ANY_SIEVE.finditer(raw_text))[:60]:
        start = max(0, match.start() - 500)
        end = min(len(raw_text), match.end() + 500)
        window = _clip_to_sentence(re.sub(r"\s+", " ", raw_text[start:end]).strip(), CHUNK_MAX_CHARS)
        if len(window) < CHUNK_MIN_CHARS:
            continue
        prefix = _fingerprint(" ".join(window.split()[:18]))
        if prefix in seen:
            continue
        score = _signal_score(window)
        if score <= 0:
            continue
        seen.add(prefix)
        out.append({"text": window, "fp": _fingerprint(window), "score": score, "heading": ""})
        used += len(window)
        if used >= char_cap:
            break
    return out


def build_chunks(raw_text: str, char_cap: int = MAX_FILTERED_CHARS) -> List[Dict[str, Any]]:
    """
    Chunking for GraphRAG: heading-prefixed, sentence-bounded, signal-ranked excerpts with
    duplicates dropped.

    The previous selector walked the document in order and kept anything matching one keyword
    until the character cap was spent, so Item 1 boilerplate consumed the budget, mid-document
    disclosures were cut off, and the same risk factor reappeared for every fiscal year.
    """
    if not raw_text:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for heading, body in _iter_paragraphs(raw_text):
        if len(body) < CHUNK_MIN_CHARS or not ANY_SIEVE.search(body):
            continue
        text = _clip_to_sentence(f"{heading}: {body}" if heading else body, CHUNK_MAX_CHARS)
        # Fingerprint the body, not the stored text: the same risk factor reappears under both
        # Item 1 and Item 1A, and a heading-relative fingerprint lets it occupy two slots.
        fp = _fingerprint(body)
        if fp in seen:
            continue
        score = _signal_score(body)
        if score <= 0:
            continue
        seen.add(fp)
        candidates.append({"text": text, "fp": fp, "score": score, "heading": heading})

    if not candidates:
        candidates = _keyword_windows(raw_text, char_cap)

    candidates.sort(key=lambda item: (-item["score"], item["fp"]))

    selected: List[Dict[str, Any]] = []
    used = 0
    for item in candidates:
        if len(selected) >= CHUNKS_PER_FILING or used + len(item["text"]) > char_cap:
            continue
        selected.append(item)
        used += len(item["text"])
    return selected


EXHIBIT_CHAR_LIMIT = 20_000   # raw cap per exhibit before sieving
EXTRA_EXHIBIT_LIMIT = 2       # partnership/press-release exhibits per 8-K


def _exhibit_text(filing) -> str:
    """
    Reads the EX-99.x press releases attached to an 8-K.

    Partnership and investment announcements are usually filed as exhibits, not in the
    8-K cover text, so skipping them loses exactly the evidence we are looking for.
    Exhibits are capped before sieving to keep the token cost bounded.
    """
    parts: List[str] = []
    try:
        attachments = list(filing.attachments)
    except Exception:
        return ""

    for attachment in attachments[:16]:
        description = (getattr(attachment, "description", "") or "").upper()
        if not description.startswith("EX-99"):
            continue
        try:
            text = str(attachment.text() or "")
        except Exception:
            continue
        if text.strip():
            parts.append(text[:EXHIBIT_CHAR_LIMIT])
        if len(parts) >= EXTRA_EXHIBIT_LIMIT:
            break
    return "\n\n".join(parts)


def extract_form_text(filing, profile: FormProfile) -> str:
    """Pulls the readable text of one filing, using section parsing for annual reports."""
    if profile.kind == "annual":
        try:
            return extract_filing_sections(filing.obj(), str(filing.form))
        except Exception as exc:
            logger.debug("Section extraction failed for %s: %s", filing.accession_number, exc)

    try:
        text = str(filing.text() or "")
    except Exception as exc:
        logger.debug("Text extraction failed for %s: %s", filing.accession_number, exc)
        text = ""

    if profile.kind == "event":
        exhibits = _exhibit_text(filing)
        if exhibits:
            text = f"{text}\n\n{exhibits}"
    return text


def normalize_entity_name(name: str) -> str:
    """Normalizes company names using canonical alias mapping."""
    clean = (name or "").strip()
    return KNOWN_ALIASES.get(clean.lower(), clean.upper())


TICKER_PATTERN = re.compile(r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?|[0-9]{4,6}\.[A-Z]{1,3}")


def looks_like_ticker(name: str) -> bool:
    """True for a stock ticker shape ('NVDA', '000660.KS'), False for prose names."""
    return bool(TICKER_PATTERN.fullmatch(name.strip().upper()))


def ensure_company(relate_graph: RelateSupplyChainGraph, raw: str) -> Optional[str]:
    """
    Resolves an extracted company to a graph node, creating it only when the name is a
    real ticker. Filings name plenty of unlisted firms and product lines; adding those
    as nodes is what turned the chart into a junk pile.
    """
    name = normalize_entity_name(raw or "")
    if not name:
        return None

    existing = relate_graph.resolve_node_id(name)
    if existing:
        info = relate_graph.nodes.get(existing, {})
        return existing if info.get("type", "Company") == "Company" else None

    if not looks_like_ticker(name):
        return None

    relate_graph.add_node(name, name, "Company")
    return name


def _existing_company(relate_graph: RelateSupplyChainGraph, raw: str) -> Optional[str]:
    """Resolves to an existing company node only - capital links never invent nodes."""
    for candidate in (normalize_entity_name(raw or ""), (raw or "").strip()):
        node_id = relate_graph.resolve_node_id(candidate)
        if node_id and relate_graph.nodes.get(node_id, {}).get("type", "Company") == "Company":
            return node_id
    return None


def _reporter_is_company(reporting_persons, company_cik) -> bool:
    """True when the 13D/13G was filed by our company (an investment), not about it."""
    if not company_cik:
        return False
    target_cik = str(company_cik).lstrip("0")
    for person in reporting_persons or []:
        person_cik = str(getattr(person, "cik", "") or "").lstrip("0")
        if person_cik and person_cik == target_cik:
            return True
    return False


def stake_from_filing(filing, company_cik, relate_graph: RelateSupplyChainGraph) -> Optional[Dict[str, Any]]:
    """
    Reads a 13D/13G as structured data (no LLM, no full-text download).

    Only filings where OUR company is the reporting person count as an investment, and
    the issuer must already be a tracked node so ownership never invents new companies.
    """
    try:
        obj = filing.obj()
    except Exception as exc:
        logger.debug("13D/13G object failed for %s: %s", filing.accession_number, exc)
        return None

    issuer = getattr(obj, "issuer_info", None)
    issuer_name = (getattr(issuer, "name", "") or "").strip()
    if not issuer_name:
        return None

    if not _reporter_is_company(getattr(obj, "reporting_persons", None), company_cik):
        return None  # an institution's stake in our company, not a company investment

    target = _existing_company(relate_graph, issuer_name)
    if not target:
        logger.debug("Stake issuer %r is not a tracked company; skipping.", issuer_name)
        return None

    return {
        "target": target,
        "issuer_name": issuer_name,
        "stake_pct": getattr(obj, "total_percent", None),
        "form": str(filing.form),
        "accession_no": str(filing.accession_number),
        "as_of": str(filing.filing_date),
    }


def _collect_documents(company, company_cik, ticker: str, relate_graph: RelateSupplyChainGraph) -> List[Dict[str, Any]]:
    """Collects unprocessed filings across all configured forms, cheapest signal first."""
    documents: List[Dict[str, Any]] = []

    for form, profile in FORM_PROFILES.items():
        try:
            filings = company.get_filings(form=form)
            candidates = list(filings)[: profile.filing_limit]
        except Exception as exc:
            logger.debug("No %s filings for %s (%s)", form, ticker, exc)
            continue

        for filing in candidates:
            accession_no = str(filing.accession_number)
            filing_key = f"{ticker}_{accession_no}"
            if relate_graph.check_filing_exists(filing_key):
                continue

            document = {
                "form": form,
                "accession_no": accession_no,
                "as_of": str(filing.filing_date),
                "profile": profile,
                "chunks": [],
                "text": "",
                "stake": None,
            }

            if profile.kind == "stake":
                document["stake"] = stake_from_filing(filing, company_cik, relate_graph)
            else:
                raw_text = extract_form_text(filing, profile)
                chunks = build_chunks(raw_text, profile.char_cap)
                document["chunks"] = chunks
                document["text"] = "\n\n".join(chunk["text"] for chunk in chunks)
                document["raw_chars"] = len(raw_text)

            documents.append(document)

    return documents


EXTRACTION_PROMPT = """You are an expert Graph Information Extraction engine for an AI supply chain knowledge graph.

The excerpt below comes from a {form} filing (accession {accession_no}) of {focus_ticker}.

Allowed node types:
- Company: ONLY publicly listed companies, written as a stock ticker (e.g. NVDA, TSM, 000660.KS)

Do not emit component, product, process or technology nodes at all
(e.g. "CoWoS", "HBM3e", "3nm", "InFo", "EMIB", "CPO", "SerDes", "liquid cooling").
This graph contains listed companies only, connected to each other.

Allowed relationships:
- (Company)-[:SUPPLIES_TO]->(Company)      one company supplies goods/services to another
- (Company)-[:INVESTED_IN]->(Company)      an equity stake / investment, include "stake_pct" if stated
- (Company)-[:PARTNERS_WITH]->(Company)    strategic partnership / joint venture / MOU

Rules:
- Only extract what the text supports. Do not use outside knowledge.
- {focus_ticker} is the filer; prefer it as the source of relationships it describes.
- Never emit a relationship whose endpoints are not in the text.
- COMPETES_WITH is not a supported type: if the text only says a company is a competitor,
  a peer or a rival, emit NO relationship at all. PARTNERS_WITH is only for an announced
  collaboration, joint venture, strategic alliance or agreement between the two companies.

Filing Text:
{filing_text}

Respond ONLY in valid JSON:
{{
  "companies": ["NVDA", "TSM"],
  "relationships": [
    {{"source": "TSM", "rel": "SUPPLIES_TO", "target": "NVDA"}},
    {{"source": "NVDA", "rel": "INVESTED_IN", "target": "ORCL", "stake_pct": null}},
    {{"source": "NVDA", "rel": "PARTNERS_WITH", "target": "ORCL", "note": "joint AI cloud build-out"}}
  ]
}}
"""


async def _extract_for_document(document: Dict[str, Any], focus_ticker: str, llm: Any, semaphore: asyncio.Semaphore):
    """One LLM call for one evidence document, under a concurrency semaphore."""
    async with semaphore:
        prompt = EXTRACTION_PROMPT.format(
            form=document["form"],
            accession_no=document["accession_no"],
            focus_ticker=focus_ticker,
            filing_text=document["text"],
        )
        try:
            response = await asyncio.to_thread(llm.invoke, prompt)
        except Exception as exc:
            logger.warning("LLM extraction failed for %s %s: %s", document["form"], document["accession_no"], exc)
            return None

    raw = getattr(response, "content", response)
    match = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        logger.warning("Unparseable extraction JSON for %s %s", document["form"], document["accession_no"])
        return None


def _apply_extraction(data: Dict[str, Any], focus_ticker: str, document: Dict[str, Any],
                      relate_graph: RelateSupplyChainGraph,
                      tracked_companies: Optional[set] = None) -> Dict[str, int]:
    """
    Applies one extraction result to the graph with per-form provenance.

    ``tracked_companies`` is the set of companies that existed before this run: capital
    and partnership links may only connect those, so a document listing many companies
    cannot quietly expand the graph (the extraction's own 'companies' array is added
    first, which used to defeat the 'existing nodes only' check).
    """
    stats = {"companies": 0, "edges": 0, "skipped": 0}
    provenance = {
        "source_form": document["form"],
        "accession_no": document["accession_no"],
        "as_of": document["as_of"],
    }
    if tracked_companies is None:
        tracked_companies = {
            node_id for node_id, info in relate_graph.nodes.items()
            if info.get("type", "Company") == "Company"
        }

    for company_name in data.get("companies", []):
        if ensure_company(relate_graph, company_name):
            stats["companies"] += 1

    for relationship in data.get("relationships", []):
        rel = str(relationship.get("rel", "")).upper()
        source_name = relationship.get("source", "")
        target_name = relationship.get("target", "")

        if rel == "SUPPLIES_TO":
            source = ensure_company(relate_graph, source_name)
            target = ensure_company(relate_graph, target_name)
        elif rel in ("INVESTED_IN", "PARTNERS_WITH"):
            # Capital and partnership links are only drawn between companies that were
            # already tracked before this run.
            source = _existing_company(relate_graph, source_name)
            target = _existing_company(relate_graph, target_name)
            if not (source in tracked_companies and target in tracked_companies):
                stats["skipped"] += 1
                continue
        else:
            # Component relationships (PRODUCES / DEPENDS_ON) are retired with the layer.
            stats["skipped"] += 1
            continue

        if not source or not target or source == target:
            stats["skipped"] += 1
            continue

        props = dict(provenance)
        if relationship.get("stake_pct"):
            props["stake_pct"] = relationship["stake_pct"]
        if relationship.get("note"):
            props["notes"] = relationship["note"]

        if relate_graph.add_edge(source, target, rel, **props):
            stats["edges"] += 1
        else:
            stats["skipped"] += 1

    return stats


def _store_chunks(relate_graph: RelateSupplyChainGraph, ticker: str, document: Dict[str, Any], embedder) -> int:
    """Embeds the retained excerpts for GraphRAG retrieval (local model, no API cost)."""
    chunks = document.get("chunks") or []
    if not chunks or embedder is None:
        return 0

    # A risk factor filed unchanged in FY2024 and FY2025 would otherwise take two of the eight
    # slots and hand the model the same paragraph twice.
    known = relate_graph.chunk_fingerprints(ticker)
    fresh = [chunk for chunk in chunks if chunk["fp"] not in known]
    if not fresh:
        return 0

    texts = [chunk["text"] for chunk in fresh]
    try:
        if hasattr(embedder, "embed_documents"):
            embeddings = embedder.embed_documents(texts)
        elif hasattr(embedder, "encode"):
            embeddings = embedder.encode(texts).tolist()
        else:
            embeddings = [embedder.embed_query(text) for text in texts]
    except Exception as exc:
        logger.warning("Embedding failed for %s %s: %s", document["form"], document["accession_no"], exc)
        return 0

    rows = [
        {
            "id": f"{ticker}_{document['accession_no']}_{chunk['fp']}",
            "text": chunk["text"],
            "ticker": ticker,
            "accession_no": document["accession_no"],
            "form": document.get("form"),
            "as_of": document.get("as_of"),
            "heading": chunk.get("heading") or "",
            "fp": chunk["fp"],
            "embedding": embedding,
        }
        for chunk, embedding in zip(fresh, embeddings)
    ]
    # one UNWIND statement for the whole filing, instead of one rewrite per excerpt
    return relate_graph.save_chunks(rows)


async def efficient_ingest_ticker(
    ticker: str,
    relate_graph: RelateSupplyChainGraph,
    llm: Any,
    embedder: Any,
    driver: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Multi-form ingestion pipeline for one ticker.

    1. Idempotency check per accession number (0 tokens when nothing is new)
    2. Collect 10-K/20-F, 8-K, 13D/13G, DEF 14A filings
    3. Filter to signal paragraphs with the local regex sieve
    4. Structured stake edges for 13D/13G (0 tokens)
    5. Concurrent LLM extraction for prose documents (bounded char budget)
    6. Local embeddings + filing records
    """
    if not HAS_EDGAR:
        return {
            "status": "error",
            "message": "edgartools is not installed or failed to initialize.",
            "tokens": 0,
        }

    init_sec_identity()
    ticker = ticker.upper().strip()
    logger.info("Starting multi-form Relate ingestion for: %s", ticker)

    try:
        company = EdgarCompany(ticker)
        company_cik = getattr(company, "cik", None)
    except Exception as exc:
        logger.error("EDGAR company lookup failed: %s", exc)
        return {"status": "error", "message": f"Failed fetching SEC company for {ticker}: {exc}", "tokens": 0}

    try:
        documents = await asyncio.to_thread(_collect_documents, company, company_cik, ticker, relate_graph)
    except Exception as exc:
        logger.exception("Filing collection failed for %s", ticker)
        return {"status": "error", "message": f"Failed reading SEC filings for {ticker}: {exc}", "tokens": 0}

    if not documents:
        logger.info("Nothing new to ingest for %s (0 tokens).", ticker)
        return {
            "status": "cached",
            "message": f"All tracked filings for {ticker} are already indexed. Skipped (0 tokens).",
            "tokens": 0,
        }

    # Budget the prose documents; stakes are handled separately without text.
    budget = MAX_TOTAL_CHARS
    llm_documents: List[Dict[str, Any]] = []
    stake_documents: List[Dict[str, Any]] = []
    forms_seen: Dict[str, int] = {}

    for document in documents:
        forms_seen[document["form"]] = forms_seen.get(document["form"], 0) + 1
        if document["profile"].kind == "stake":
            stake_documents.append(document)
            continue

        text_length = len(document["text"])
        take = min(text_length, budget)
        if take < MIN_SIGNAL_CHARS:
            logger.info("%s %s had no supply-chain/ownership signal; skipping LLM.",
                        document["form"], document["accession_no"])
            relate_graph.record_filing(
                f"{ticker}_{document['accession_no']}", ticker, document["accession_no"],
                document["form"], document["as_of"],
            )
            continue
        document["text"] = document["text"][:take]
        budget -= take
        llm_documents.append(document)

    estimated_tokens = sum(len(d["text"]) for d in llm_documents) // 4

    # Concurrent LLM extraction for the prose documents.
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    results = await asyncio.gather(*[
        _extract_for_document(document, ticker, llm, semaphore) for document in llm_documents
    ], return_exceptions=True)

    totals = {"companies": 0, "edges": 0, "skipped": 0}
    chunks_stored = 0
    tracked_companies = {
        node_id for node_id, info in relate_graph.nodes.items()
        if info.get("type", "Company") == "Company"
    }

    for document, result in zip(llm_documents, results):
        if isinstance(result, Exception):
            logger.warning("Extraction error for %s %s: %s",
                           document["form"], document["accession_no"], result)
        elif result:
            stats = _apply_extraction(result, ticker, document, relate_graph, tracked_companies)
            for key in totals:
                totals[key] += stats[key]
        chunks_stored += _store_chunks(relate_graph, ticker, document, embedder)
        relate_graph.record_filing(
            f"{ticker}_{document['accession_no']}", ticker, document["accession_no"],
            document["form"], document["as_of"],
        )

    # Zero-token ownership edges from structured 13D/13G metadata.
    stake_edges = 0
    for document in stake_documents:
        stake = document["stake"]
        if stake and relate_graph.add_edge(
            ticker, stake["target"], "INVESTED_IN",
            source_form=stake["form"],
            accession_no=stake["accession_no"],
            as_of=stake["as_of"],
            stake_pct=stake["stake_pct"],
            notes=f"{ticker} stake in {stake['issuer_name']}",
        ):
            stake_edges += 1
            logger.info("Stake edge: %s -[INVESTED_IN]-> %s", ticker, stake["target"])
        relate_graph.record_filing(
            f"{ticker}_{document['accession_no']}", ticker, document["accession_no"],
            document["form"], document["as_of"],
        )

    relate_graph._persist()

    forms_label = ", ".join(f"{form} x{count}" for form, count in sorted(forms_seen.items()))
    edge_count = totals["edges"] + stake_edges
    message = (
        f"Read {len(documents)} filing(s) ({forms_label}). "
        f"Added {edge_count} relationship(s) and {chunks_stored} GraphRAG chunk(s). "
        f"~{estimated_tokens:,} tokens."
    )
    logger.info("Ingestion complete for %s: %s", ticker, message)

    return {
        "status": "success",
        "message": message,
        "tokens": estimated_tokens,
        "forms": forms_seen,
        "filings": len(documents),
        "edges_added": edge_count,
        "stake_edges": stake_edges,
        "chunks": chunks_stored,
    }
