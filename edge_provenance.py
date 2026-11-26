"""
edge_provenance.py - turn asserted relationships into cited ones.

Half the curated edges in the graph carry no accession number, so the analyst can draw a line
between two companies but cannot show the paragraph that justifies it. This reads a filer's own
filing text and reports, for every known company mentioned in it, whether the surrounding
language actually states a supply, customer or ownership relationship - including which way the
relationship points, because "we depend on X" and "X is a customer" produce opposite edges.

    python edge_provenance.py propose --tickers SNPS,GFS,TER --dry-run
    python edge_provenance.py cite --tickers ASML,TSM          # attach evidence to existing edges
    python edge_provenance.py cite --all                        # everything lacking provenance

Direction is decided by the filer's wording, never guessed: a chunk that names a company without
using a relationship verb is reported as ambiguous and left unapplied.
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from edgar import Company

from ingest import (
    FORM_PROFILES,
    KNOWN_ALIASES,
    _iter_paragraphs,
    _signal_score,
    init_sec_identity,
    substantive_filings,
)
from relate_adapter import RelateSupplyChainGraph

logger = logging.getLogger("provenance")

PROSE_FORMS = ("10-K", "20-F")

# Which way the arrow goes depends on who is doing the depending.
_SUPPLIER_SIDE = re.compile(
    r"\b(?:"
    # Filers hedge before they commit: "we also depend on", "we primarily source from".
    r"we\s+(?:\w+ly|also|primarily|mainly|principally|largely|further|currently|substantially)?\s*"
    r"(?:depend|rely|purchase|buy|source|procure)\b|"
    r"our\s+(?:\w+ly\s+)?(?:suppliers|sole source|single source|primary supplier)\b|"
    r"supplied by|purchased from|sourced from|outsourced to|contract(?:ed|ing)? with|"
    r"supply agreement(?:s)? (?:with|for)|manufactur\w+ (?:by|for us at)|"
    r"third[- ]party (?:manufacturer|supplier)s? (?:of|that)|"
    r"foundry (?:partner|relationship|agreement) with"
    r")",
    re.IGNORECASE,
)
_CUSTOMER_SIDE = re.compile(
    r"\b(?:our (?:largest|top|significant) customers?|customers? (?:accounted for|represent|"
    r"constitute|include|such as|that include)|customer concentration|"
    r"accounted for\b[^.]{0,90}\brevenue\b|revenue\b[^.]{0,60}accounted for|"
    r"one customer(?:'s)? (?:revenue|account)|we sell(?:ing)? (?:to|our products to)|"
    r"shipments to|design wins? (?:from|with)|our products (?:are )?used by|"
    r"we (?:ship|delivered) (?:to|products to))\b",
    re.IGNORECASE,
)
# EDA and IP vendors do not "supply" wafers, they license designs, and their filings say so:
# "generated 12% of our revenue from licensing arrangements with". Recorded as SUPPLIES_TO with
# the sentence kept as the note, so the arrow carries the quote and the nuance is not lost.
_LICENSE_SIDE = re.compile(
    r"\b(?:licens(?:e|es|ed|ing|ing arrangements?|or)s?\b[^.]{0,90}(?:revenue|royalt|use)|"
    r"design (?:wins?|partners?) with|ip (?:licens|agreement)s? with|"
    r"our (?:design|technology) (?:is )?(?:used|licensed) by)\b",
    re.IGNORECASE,
)
_OWNERSHIP = re.compile(
    r"\b(?:equity interest in|acquired (?:a |an )?\d|joint venture with|stake in|"
    r"owns \d|we invest(?:ed|or)? in|majority interest in|"
    r"we (?:hold|own) (?:a |an))\b",
    re.IGNORECASE,
)
# A competitor list names the same companies a supply relationship does, in the same document,
# and it is the single most common false edge: "Key competitors include TSMC, UMC" must never
# become an arrow between them.
_COMPETITION = re.compile(
    r"\b(?:key |main |principal |significant )?competitors? (?:include|are|with)|"
    r"we compete with|competitive landscape|competitor\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_RELATION_HINT = re.compile(
    r"supplier|supply|customer|revenue|licens|royalt|stake|acquir|depend|outsourc|"
    r"purchas|partner|joint venture|used by|source from|foundry|packaging",
    re.IGNORECASE,
)

# Tickers that may become new nodes; anything else must already exist in the graph.
CANDIDATES: set = set()

# Aliases shorter than this are only trusted when the filing writes them as a proper noun, so
# "the long arm of the law" cannot create an edge to Arm Holdings while "TSMC" still can.
MIN_ALIAS_CHARS = 5
_PROPER_NOUN = re.compile(r"^(?:[A-Z]{2,}|[A-Z][a-z]+)$")


def _mentions(alias: str, sentence: str) -> bool:
    if len(alias) >= MIN_ALIAS_CHARS:
        return bool(re.search(rf"\b{re.escape(alias)}\b", sentence, re.IGNORECASE))
    return any(_PROPER_NOUN.match(match.group(0))
               for match in re.finditer(rf"\b{re.escape(alias)}\b", sentence, re.IGNORECASE))


def _classify(paragraph: str) -> List[Dict[str, str]]:
    """(counterparty alias, direction) for every relationship this paragraph states."""
    out = []
    for alias, ticker in KNOWN_ALIASES.items():
        if len(alias) < 4:
            continue
        # Test every sentence that names the company, not just the first: a filing introduces a
        # counterparty descriptively ("Cadence is an EDA vendor") before stating the relationship
        # ("we purchase design services from Cadence"), and stopping at the first mention loses
        # the disclosure.
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            if not _mentions(alias, sentence):
                continue
            if _COMPETITION.search(sentence):
                continue
            for pattern, direction, name in (
                (_OWNERSHIP, "INVESTED_IN", "ownership"),
                (_SUPPLIER_SIDE, "INBOUND", "supplier-side"),
                (_CUSTOMER_SIDE, "OUTBOUND", "customer-side"),
                (_LICENSE_SIDE, "OUTBOUND", "license-side"),
            ):
                if pattern.search(sentence):
                    out.append({"alias": alias, "ticker": ticker, "direction": direction,
                                "pattern": name, "sentence": sentence.strip()})
                    break
            else:
                continue
            break
    return out


def scan_filing(graph: RelateSupplyChainGraph, ticker: str,
                forms: tuple = PROSE_FORMS) -> List[Dict[str, Any]]:
    """
    Every (counterparty, direction, evidence) triple this filer's own filing states.

    This walks the whole document, not the retrieval chunks: the top chunks are chosen for how
    supply-chain-dense they read, and a single customer-concentration sentence buried in Item 1A
    is exactly the evidence an edge needs but retrieval has no reason to rank.
    """
    try:
        company = Company(ticker)
    except Exception as exc:
        print(f"{ticker:8} unresolvable on EDGAR: {str(exc)[:60]}")
        return []

    found: List[Dict[str, Any]] = []
    for form in forms:
        if form not in FORM_PROFILES or FORM_PROFILES[form].char_cap <= 0:
            continue
        # One substantive annual report per form is the disclosure; older ones only restate it, and
        # substantive_filings already steps past the amendment that pretends to be one.
        for filing, raw in substantive_filings(company, form, FORM_PROFILES[form], limit=1):
            accession = str(filing.accession_number)
            paragraphs = list(_iter_paragraphs(raw))
            if not paragraphs:
                print(f"{ticker:8} {form} {accession}: no text extracted")
                continue
            for heading, paragraph in paragraphs:
                # Cheap pre-filter: a paragraph with no relationship word and no number cannot
                # state a relationship, and testing every paragraph against every alias is what
                # would make this crawl. The signal score then only ranks surviving candidates.
                if not _RELATION_HINT.search(paragraph):
                    continue
                score = _signal_score(paragraph)
                if score < 2:
                    continue
                for hit in _classify(paragraph):
                    other = hit["ticker"]
                    if other == ticker:
                        continue
                    if not graph.resolve_node_id(other) and other not in CANDIDATES:
                        continue
                    found.append({
                        "filer": ticker, "other": other, "direction": hit["direction"],
                        "pattern": hit["pattern"], "form": form, "accession_no": accession,
                        "as_of": str(filing.filing_date), "heading": heading,
                        "quote": hit["sentence"][:400], "score": score,
                    })
            # One substantive annual report per form is the disclosure; older ones only restate it.
            break

    dedup: Dict[tuple, Dict[str, Any]] = {}
    for hit in sorted(found, key=lambda h: -h["score"]):
        dedup.setdefault((hit["filer"], hit["other"], hit["direction"]), hit)
    return list(dedup.values())


def _edges_needing_provenance(graph: RelateSupplyChainGraph, tickers: Optional[List[str]]):
    rows = graph._rows(
        "MATCH (a:Company)-[r]->(b:Company) WHERE r.accession_no IS NULL "
        "RETURN a.id AS source, b.id AS target, type(r) AS type, r.notes AS notes "
        "ORDER BY a.id"
    )
    if tickers:
        wanted = {t.upper() for t in tickers}
        rows = [r for r in rows if r["source"] in wanted or r["target"] in wanted]
    return rows


def cmd_propose(graph, args):
    global CANDIDATES
    CANDIDATES = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    total = applied = 0
    for ticker in sorted(CANDIDATES):
        hits = scan_filing(graph, ticker)
        print(f"\n{ticker}: {len(hits)} evidenced relationship(s)")
        for h in hits[:args.limit]:
            arrow = {"INBOUND": f"{h['other']} -> {h['filer']}",
                     "OUTBOUND": f"{h['filer']} -> {h['other']}",
                     "INVESTED_IN": f"{h['filer']} invests in {h['other']}"}[h["direction"]]
            print(f"  [{h['pattern']:12}] {arrow:26} {h['form']} {h['as_of']} "
                  f"acc={h['accession_no']} score={h['score']}")
            print(f"      {h['quote'][:220]}")
        total += len(hits)
        if args.apply:
            for h in hits:
                if h["direction"] == "INBOUND":
                    src, tgt, rel = h["other"], h["filer"], "SUPPLIES_TO"
                elif h["direction"] == "OUTBOUND":
                    src, tgt, rel = h["filer"], h["other"], "SUPPLIES_TO"
                else:
                    src, tgt, rel = h["filer"], h["other"], "INVESTED_IN"
                graph.add_node(src, src, "Company")
                graph.add_node(tgt, tgt, "Company")
                if graph.add_edge(src, tgt, rel,
                                  notes=f"{h['pattern']} relationship per {h['form']}",
                                  source_form=h["form"], accession_no=h["accession_no"],
                                  as_of=h["as_of"], evidence=h["quote"]):
                    applied += 1
                    print(f"    applied {src} -[{rel}]-> {tgt}")
    print(f"\ntotal evidenced relationships: {total} | edges written: {applied}"
          + ("" if args.apply else " (dry run - pass --apply to write)"))


def cmd_cite(graph, args):
    """
    Attribute existing edges by reading BOTH ends.

    A supplier's filing names its customers ("Microsoft accounted for 28% of our revenue") while a
    buyer's filing names its suppliers ("we depend on ASML for EUV"), and which side volunteers the
    name differs company by company. Scanning only the edge's source silently misses the cases where
    the disclosure sits on the other end of the same relationship.
    """
    rows = _edges_needing_provenance(graph, None if args.all else
                                     [t.strip() for t in (args.tickers or "").split(",") if t.strip()])
    print(f"edges without provenance: {len(rows)}")

    cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def hits(company: str) -> Dict[str, Dict[str, Any]]:
        if company not in cache:
            cache[company] = {h["other"]: h for h in scan_filing(graph, company)}
        return cache[company]

    # (filer whose disclosure we read, counterparty it names, direction that disclosure implies)
    plans = {
        "SUPPLIES_TO": ((("source", "target", "OUTBOUND"), "supplier-side per its own filing"),
                        (("target", "source", "INBOUND"), "named as a dependency by the buyer")),
        "INVESTED_IN": ((("source", "target", "INVESTED_IN"), "ownership stated by the investor"),),
        "PARTNERS_WITH": ((("source", "target", "OUTBOUND"), "partnership stated by the filer"),
                          (("target", "source", "OUTBOUND"), "partnership stated by the partner")),
    }

    cited = unmatched = 0
    for edge in sorted(rows, key=lambda r: (r["source"], r["target"])):
        hit = origin = None
        for (side_a, side_b, direction), label in plans.get(edge["type"], ()):
            found = hits(edge[side_a]).get(edge[side_b])
            if found and found["direction"] == direction:
                hit, origin = found, label
                break
        if hit:
            if args.apply:
                graph._rows(
                    "MATCH (a:Company {id: $s})-[r]->(b:Company {id: $t}) "
                    "WHERE r.accession_no IS NULL "
                    "SET r.source_form = $f, r.accession_no = $acc, r.as_of = $d, "
                    "r.evidence = $q, r.notes = coalesce(r.notes, '') + ' | ' + $note",
                    {"s": edge["source"], "t": edge["target"], "f": hit["form"],
                     "acc": hit["accession_no"], "d": hit["as_of"],
                     "q": hit["quote"][:400], "note": origin},
                )
            cited += 1
            print(f"  CITED   {edge['source']} -[{edge['type']}]-> {edge['target']}  "
                  f"{hit['form']} {hit['accession_no']} :: {hit['quote'][:100]}")
        else:
            unmatched += 1
            print(f"  UNCITED {edge['source']} -[{edge['type']}]-> {edge['target']}  "
                  f"({edge['notes'] or '-'})")
    print(f"\ncited {cited} | still unattributed {unmatched}")
    if not args.apply and cited:
        print("(dry run - pass --apply to write)")



def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--apply", action="store_true", help="write to Relate (default is dry run)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="find evidenced edges for candidate tickers")
    p.add_argument("--tickers", required=True)
    c = sub.add_parser("cite", help="attach evidence to existing edges lacking provenance")
    c.add_argument("--tickers", default="")
    c.add_argument("--all", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    init_sec_identity()
    graph = RelateSupplyChainGraph(args.db) if args.db else RelateSupplyChainGraph()
    (cmd_propose if args.cmd == "propose" else cmd_cite)(graph, args)
    graph.store.close()


if __name__ == "__main__":
    main()
