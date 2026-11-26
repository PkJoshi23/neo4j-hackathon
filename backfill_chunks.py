"""
backfill_chunks.py - give the graph the filing text it is missing.

Ingestion skips any filing already recorded in Relate, so tickers registered before the
embedding step worked have Filing nodes and no Chunk nodes: the analyst can name a supplier
edge but can never show the paragraph that supports it. This re-reads exactly those filings
from EDGAR (cache-first under .edgar_data), chunks them and embeds them with the local model,
then writes (:Chunk)-[:ABOUT]->(:Company) behind the same indexes the app queries. When a
recorded annual report turns out to hold no document at all - an amendment filed under the same
form name - it is re-pointed at the report that does, so the ticker stops reporting as pending.

No LLM is imported, so this costs no DeepSeek tokens.

    python backfill_chunks.py --dry-run
    python backfill_chunks.py --tickers NVDA,MU
    python backfill_chunks.py
"""

import argparse
import logging
import time
from typing import Dict, List

from edgar import Company
from sentence_transformers import SentenceTransformer

from ingest import (
    FORM_PROFILES,
    build_chunks,
    extract_form_text,
    init_sec_identity,
    substantive_filings,
)
from relate_adapter import CHUNK_EMBEDDING_DIM, RelateSupplyChainGraph

logger = logging.getLogger("backfill")

# Below this the "filing" is edgartools' repr of an empty document, not a document.
EMPTY_DOC_CHARS = 500


def filings_without_chunks(graph: RelateSupplyChainGraph) -> Dict[str, List[Dict]]:
    """Filing nodes that have no Chunk carrying the same accession number, grouped by ticker."""
    covered = {
        row["acc"] for row in graph._rows("MATCH (c:Chunk) RETURN DISTINCT c.accession_no AS acc")
        if row.get("acc")
    }
    rows = graph._rows(
        "MATCH (f:Filing) RETURN f.ticker AS ticker, f.accession_no AS accession_no, "
        "f.form AS form, f.filing_date AS filing_date ORDER BY ticker, filing_date DESC"
    )
    pending: Dict[str, List[Dict]] = {}
    for row in rows:
        if not row.get("accession_no") or row["accession_no"] in covered:
            continue
        if row["form"] not in FORM_PROFILES or FORM_PROFILES[row["form"]].char_cap <= 0:
            continue  # structured stake filings have no prose to embed
        pending.setdefault(row["ticker"], []).append(row)
    return pending


def _match_filing(company, form: str, accession_no: str):
    try:
        for filing in company.get_filings(form=form):
            if str(filing.accession_number) == accession_no:
                return filing
    except Exception as exc:
        logger.warning("  %s %s lookup failed: %s", company.ticker, form, exc)
    return None


def backfill(graph: RelateSupplyChainGraph, model, pending: Dict[str, List[Dict]],
             dry_run: bool = False) -> Dict[str, int]:
    per_ticker: Dict[str, int] = {}
    started = time.perf_counter()

    for ticker, filings in sorted(pending.items()):
        if dry_run:
            print(f"{ticker:10} {len(filings):2} filings without chunks")
            continue
        try:
            company = Company(ticker)
        except Exception as exc:
            print(f"{ticker:10} SKIPPED - company not resolvable ({str(exc)[:60]})")
            continue

        stored = 0
        for filing_row in filings:
            profile = FORM_PROFILES[filing_row["form"]]
            filing = _match_filing(company, filing_row["form"], filing_row["accession_no"])
            raw = "" if filing is None else extract_form_text(filing, profile)
            chunks = build_chunks(raw, profile.char_cap) if raw else []
            accession_no, as_of = filing_row["accession_no"], filing_row["filing_date"]

            if not chunks and profile.kind == "annual" and len(raw) < EMPTY_DOC_CHARS:
                # The graph recorded an accession with no prose behind it - an annual-report
                # amendment whose primary document is a signature page. Move the Filing node onto
                # the report that does have text, or this ticker stays permanently unquotable.
                alt = substantive_filings(company, filing_row["form"], profile, limit=1)
                if not alt:
                    print(f"{ticker:10} {accession_no} empty ({len(raw)} chars) and no substitute")
                    continue
                filing, raw = alt[0]
                chunks = build_chunks(raw, profile.char_cap)
                accession_no, as_of = str(filing.accession_number), str(filing.filing_date)
                graph.record_filing(f"{ticker}_{accession_no}", ticker, accession_no,
                                    filing_row["form"], as_of)
                graph._rows(
                    "MATCH (f:Filing {id: $id}) SET f.accession_no = $acc, f.filing_date = $d",
                    {"id": f"{ticker}_{filing_row['accession_no']}", "acc": accession_no,
                     "d": as_of},
                )
                print(f"{ticker:10} {filing_row['accession_no']} held no document "
                      f"-> re-pointed at {accession_no}")

            if not chunks:
                print(f"{ticker:10} {accession_no} no signal paragraphs matched")
                continue

            vectors = model.encode(
                [chunk["text"] for chunk in chunks], batch_size=16, convert_to_numpy=True
            ).tolist()
            rows = [
                {
                    "id": f"{ticker}_{accession_no}_{chunk['fp']}",
                    "text": chunk["text"],
                    "ticker": ticker,
                    "accession_no": accession_no,
                    "form": filing_row["form"],
                    "as_of": as_of,
                    "heading": chunk["heading"],
                    "fp": chunk["fp"],
                    "embedding": vector,
                }
                for chunk, vector in zip(chunks, vectors)
                if len(vector) == CHUNK_EMBEDDING_DIM
            ]
            stored += graph.save_chunks(rows)
        per_ticker[ticker] = stored
        print(f"{ticker:10} {stored:4} chunks from {len(filings)} filings")

    if not dry_run:
        print(f"\nelapsed {time.perf_counter() - started:.0f}s | "
              f"chunks now in Relate: {graph.chunk_count()}")
    return per_ticker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Relate file (defaults to the app's store)")
    parser.add_argument("--tickers", default=None, help="comma-separated subset, e.g. NVDA,MU")
    parser.add_argument("--all", action="store_true",
                        help="re-chunk filings that already have chunks too")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    init_sec_identity()

    graph = RelateSupplyChainGraph(args.db) if args.db else RelateSupplyChainGraph()
    pending = filings_without_chunks(graph)
    if args.all:
        rows = graph._rows(
            "MATCH (f:Filing) RETURN f.ticker AS ticker, f.accession_no AS accession_no, "
            "f.form AS form, f.filing_date AS filing_date ORDER BY ticker"
        )
        pending = {}
        for row in rows:
            profile = FORM_PROFILES.get(row["form"])
            if profile and profile.char_cap > 0:
                pending.setdefault(row["ticker"], []).append(row)
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        pending = {t: f for t, f in pending.items() if t in wanted}

    print(f"tickers without full chunk coverage: {len(pending)} | filings to read: "
          f"{sum(len(v) for v in pending.values())}")
    if not pending:
        print("nothing to do - every filing already has excerpts in Relate")
        return

    model = None if args.dry_run else SentenceTransformer("BAAI/bge-small-en-v1.5")
    backfill(graph, model, pending, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
