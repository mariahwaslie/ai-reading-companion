"""Summaries over result rows, shared by run_eval.py and report.py.

Standard library only (plus eval.metrics), so report.py can run on any Python
that has matplotlib, without the app's dependencies.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from eval import metrics

DEPTH = 15  # Agent.run sends 15 chunks to the model
SUMMARY_KS = (1, 3, 5, DEPTH)
PRECISION_KS = (5, DEPTH)


def _bool(v):
    if v is None or v == "":
        return None
    return v if isinstance(v, bool) else v == "True"


def _int(v):
    return None if v is None or v == "" else int(v)


def _float(v):
    return None if v is None or v == "" else float(v)


def _ids(v):
    return [int(x) for x in v.split()] if v else []


def load_rows(path):
    """Read a results CSV, converting the columns the summaries use back to Python types."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["answerable"] = _bool(r["answerable"])
        r["first_gold_rank"] = _int(r.get("first_gold_rank"))
        r["exact_gold_rank"] = _int(r.get("exact_gold_rank"))
        r["faithfulness"] = _float(r.get("faithfulness"))
        r["relevance"] = _int(r.get("relevance"))
        r["abstained"] = _bool(r.get("abstained"))
        r["gold_in_context"] = _bool(r.get("gold_in_context"))
        r["needs_review"] = _bool(r.get("needs_review"))
        add_derived(r)
    return rows


def add_derived(r):
    """Precision@k and citation accuracy, recomputed from the raw columns so every CSV is scored alike.

    Precision uses the longest retrieval list the CSV saved: older runs kept only their top 5, so
    their precision@15 is None. Citation accuracy needs evidence_chunk_ids, which run_eval writes
    (and --backfill adds to older CSVs).
    """
    gold = _ids(r.get("gold_chunk_ids"))
    retrieved = _ids(r.get("retrieved_ids") or r.get("context_chunk_ids") or r.get("retrieved_top5"))
    for k in PRECISION_KS:
        r[f"precision@{k}"] = metrics.precision_at_k(retrieved, gold, k) if r["answerable"] else None
    r["cited_chunk_ids"] = metrics.cited_chunk_ids(r.get("answer"))
    evidence = r.get("evidence_chunk_ids")
    r["citation_accuracy"] = (metrics.citation_accuracy(r["cited_chunk_ids"], _ids(evidence))
                              if r["answerable"] and evidence else None)


def load_run(manifest_path):
    """Return (manifest, [rows per repeat]) for a run written by run_eval.py."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    folder = Path(manifest_path).parent
    return manifest, [load_rows(folder / name) for name in manifest["csv_files"]]


def hit_curve(rows, max_k=DEPTH):
    ranks = [r["first_gold_rank"] for r in rows if r["answerable"]]
    return [mean(metrics.hit_at_k(rk, k) for rk in ranks) for k in range(1, max_k + 1)] if ranks else []


def summarize(rows, depth=DEPTH):
    """Headline metrics for one run (or a subset of its questions)."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    s = {"questions": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable)}
    if answerable:
        ranks = [r["first_gold_rank"] for r in answerable]
        for k in SUMMARY_KS:
            s[f"hit@{k}"] = mean(metrics.hit_at_k(rk, k) for rk in ranks)
        s["mrr"] = mean(metrics.reciprocal_rank(rk, depth) for rk in ranks)
        for k in PRECISION_KS:
            values = [r[f"precision@{k}"] for r in answerable if r.get(f"precision@{k}") is not None]
            if len(values) == len(answerable):  # skip runs that didn't save a long enough list
                s[f"precision@{k}"] = mean(values)
                s[f"precision_ceiling@{k}"] = mean(metrics.precision_ceiling(_ids(r["gold_chunk_ids"]), k)
                                                    for r in answerable)
    if any(r.get("answer") for r in rows):
        s["passes"] = sum(1 for r in rows if r.get("status") == metrics.PASS)
        s["pass_rate"] = s["passes"] / len(rows)
        faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        s["faithfulness"] = mean(faith) if faith else None
        if answerable:
            # Mean over answerable questions whose answer cites at least one chunk.
            accuracy = [r["citation_accuracy"] for r in answerable if r.get("citation_accuracy") is not None]
            if any(r.get("evidence_chunk_ids") for r in answerable):
                s["citation_accuracy"] = mean(accuracy) if accuracy else None
                s["uncited"] = sum(1 for r in answerable if not r.get("cited_chunk_ids"))
            s["relevance"] = mean(r["relevance"] for r in answerable)
            counts = Counter(r["correctness"] for r in answerable)
            for c in ("correct", "partial", "incorrect"):
                s[c] = counts[c]
        if unanswerable:
            s["refusal_rate"] = mean(1 if r["abstained"] else 0 for r in unanswerable)
    return s


def spread(summaries, key):
    """(mean, min, max) of one metric across repeats, or None if it's missing."""
    values = [s[key] for s in summaries if s.get(key) is not None]
    return (mean(values), min(values), max(values)) if values else None


def subset(rows, ids):
    wanted = set(ids)
    return [r for r in rows if r["id"] in wanted]


def groups_for(golden, rows):
    """{group name: question ids present in the run}, plus an 'all' group."""
    present = {r["id"] for r in rows}
    groups = {}
    for name, ids in golden.get("meta", {}).get("groups", {}).items():
        ids = [i for i in ids if i in present]
        if ids:
            groups[name] = ids
    if len(groups) != 1:
        groups[f"all {len(present)}"] = [r["id"] for r in rows]
    return groups


def per_question(runs):
    """Per-question view across repeats: statuses, majority outcome and pass/fail flips."""
    out = {}
    for rows in runs:
        for r in rows:
            q = out.setdefault(r["id"], {"row": r, "rows": [], "statuses": [], "faithfulness": [],
                                         "correctness": [], "abstained": [], "reasons": [], "citations": []})
            q["rows"].append(r)
            q["citations"].append(r.get("citation_accuracy"))
            q["statuses"].append(r.get("status") or "")
            q["faithfulness"].append(r.get("faithfulness"))
            q["correctness"].append(r.get("correctness") or "")
            q["abstained"].append(r.get("abstained"))
            if r.get("failure_reason"):
                q["reasons"].append(r["failure_reason"])
    for q in out.values():
        q["majority"] = Counter(q["statuses"]).most_common(1)[0][0]
        q["flipped"] = len({s == metrics.PASS for s in q["statuses"]}) > 1
    return out


def format_spread(values, digits=2, suffix=""):
    """'0.47' when every repeat agrees, else '0.89 (0.86–0.91)'."""
    if values is None:
        return "-"
    avg, lo, hi = values
    if lo == hi:
        return f"{avg:.{digits}f}{suffix}"
    return f"{avg:.{digits}f}{suffix} ({lo:.{digits}f}–{hi:.{digits}f})"
