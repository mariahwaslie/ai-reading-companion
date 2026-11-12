"""Evaluate the RAG pipeline against eval/golden_set.json.

Run from the repo root:
    python -m eval.run_eval                         # retrieval + full pipeline + LLM judge
    python -m eval.run_eval --repeats 3 --label "my change"   # repeat to see run-to-run noise
    python -m eval.run_eval --retrieval-only        # retriever only, no API calls
    python -m eval.run_eval --rescore eval/results/results_X.csv   # re-judge saved answers
    python -m eval.run_eval --show-chunks 47,74     # print chunks to check golden entries
    python -m eval.run_eval --resume eval/results/results_X.json    # finish an interrupted run
    python -m eval.run_eval --backfill eval/results/results_X.csv   # add newer metric columns to old CSVs

Retrieval experiments (eval only; the app is unchanged):
    python -m eval.run_eval --chunk-size 1000 --chunk-overlap 150 --remap-golden eval/results/golden_set_1000_150.json
    python -m eval.run_eval --retrieval-only --chunk-size 1000 --chunk-overlap 150 --golden eval/results/golden_set_1000_150.json
    python -m eval.run_eval --retrieval-only --rerank          # cross-encoder over the top 50, keep 15
    python -m eval.run_eval --retrieval-only --hybrid          # BM25 + vectors, reciprocal rank fusion

Each question's row is written to the CSV as soon as it is scored, so an interrupted run (for example,
an API quota error) keeps its finished questions and --resume picks up where it stopped.

The app's own code (parser, chunker, VectorStore, Agent) is used unchanged. The one difference is
that the nearest-neighbour lookup under VectorStore.search is exact here (see ExactSearch), so
retrieval is the same in every run; pass --approximate-search to use the app's Chroma index as-is.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval import metrics, summary  # noqa: E402
from eval.judge import Judge  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
RETRIEVAL_KS = (1, 3, 5)
# Experiments only: the app's chunker uses ["\n\n", "\n", ". "], which can leave chunks over chunk_size
# when a passage has none of them; these fall back to spaces, then characters.
FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
# Options that shape a run; saved in its manifest so --resume continues with exactly the same settings.
RUN_OPTIONS = ("golden", "book", "retrieval_only", "repeats", "label", "judge_model", "depth", "ids",
               "gold_from_evidence", "approximate_search", "chunk_size", "chunk_overlap", "rerank",
               "reranker_model", "candidates", "hybrid", "rrf_k")
# For BM25 in --hybrid: common English function words, dropped from chunks and questions alike.
STOPWORDS = frozenset("""a about after all also am an and any are as at be been but by can could did do does
for from had has have he her him his how i if in into is it its me my no not of on or our she so some such than
that the their them then there these they this those to too us was we were what when where which while who whom
why will with would you your""".split())
PRECISION_KS = (5, summary.DEPTH)
DEFAULT_JUDGE_MODEL = "gpt-4.1"

CSV_COLUMNS = [
    "id", "answerable", "question", "gold_chunk_ids", "retrieved_top5", "retrieved_ids", "first_gold_rank",
    "exact_gold_rank", "hit@1", "hit@3", "hit@5", "reciprocal_rank", "precision@5", "precision@15",
    "gold_in_context", "context_chunk_ids", "answer", "evidence_chunk_ids", "cited_chunk_ids", "citation_accuracy",
    "faithfulness", "claims_supported", "claims_total", "unsupported_claims",
    "relevance", "correctness", "abstained", "status", "failure_reason", "notes",
    "judge_reasoning", "needs_review",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.json")
    p.add_argument("--book", type=Path, help="book file (default: the one named in the golden set)")
    p.add_argument("--retrieval-only", action="store_true", help="skip generation and judging (no API calls)")
    p.add_argument("--repeats", type=int, default=1, help="run the full eval this many times (default 1)")
    p.add_argument("--label", default="", help="name for this run in the HTML report")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--depth", type=int, default=summary.DEPTH,
                   help="chunks the app sends to the model; used for MRR and hit@depth")
    p.add_argument("--ids", help="comma-separated entry ids to run, e.g. q01,q18")
    p.add_argument("--rescore", type=Path, metavar="CSV",
                   help="re-run the faithfulness judge on a results CSV's saved answers (no new answers)")
    p.add_argument("--gold-from-evidence", action="store_true",
                   help="find gold chunks by their evidence quotes instead of the stored chunk_ids "
                        "(use when trying other chunk sizes)")
    p.add_argument("--approximate-search", action="store_true",
                   help="use the app's Chroma (HNSW) index as-is instead of an exact search; retrieval can "
                        "then differ slightly between runs, as it does between uploads in the app")
    p.add_argument("--backfill", type=Path, nargs="+", metavar="CSV",
                   help="add the precision@k and citation columns to saved results CSVs, in place (no API calls)")
    p.add_argument("--show-chunks", help="comma-separated chunk ids to print, then exit")
    p.add_argument("--resume", type=Path, metavar="MANIFEST",
                   help="finish an interrupted run, reusing its saved settings and skipping saved questions")
    x = p.add_argument_group("retrieval experiments (eval only)")
    x.add_argument("--chunk-size", type=int, help="chunk with this size instead of the app's chunker, using "
                                                  "fallback separators " + repr(FALLBACK_SEPARATORS))
    x.add_argument("--chunk-overlap", type=int, help="overlap for --chunk-size")
    x.add_argument("--remap-golden", type=Path, metavar="OUT",
                   help="write a copy of the golden set with chunk_ids found from the evidence quotes "
                        "under the current chunking, then exit (the original file is not changed)")
    x.add_argument("--rerank", action="store_true",
                   help="re-rank the first-stage candidates with a cross-encoder and keep the top --depth")
    x.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    x.add_argument("--candidates", type=int, default=50, help="first-stage chunks to re-rank (default 50)")
    x.add_argument("--hybrid", action="store_true",
                   help="fuse BM25 keyword ranking with the vector ranking by reciprocal rank fusion "
                        "(needs `pip install rank-bm25`)")
    x.add_argument("--rrf-k", type=int, default=60, help="reciprocal rank fusion constant (default 60)")
    args = p.parse_args()
    if args.resume:
        if args.backfill or args.rescore or args.remap_golden or args.show_chunks:
            p.error("--resume can't be combined with --backfill, --rescore, --remap-golden or --show-chunks")
        restore_options(args, p)
    if args.hybrid and args.rerank:
        p.error("--hybrid and --rerank are separate experiments; use one at a time")
    if (args.chunk_size is None) != (args.chunk_overlap is None):
        p.error("--chunk-size and --chunk-overlap go together")
    if args.depth < max(RETRIEVAL_KS):
        p.error(f"--depth must be at least {max(RETRIEVAL_KS)}")
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    return args


def restore_options(args, parser):
    """Load an interrupted run's manifest and copy its saved run options onto args."""
    if not args.resume.exists():
        parser.error(f"--resume: {args.resume} not found")
    manifest = json.loads(args.resume.read_text(encoding="utf-8"))
    if manifest.get("complete", True):
        parser.error(f"{args.resume.name} is already complete; nothing to resume")
    for key, value in manifest["options"].items():
        setattr(args, key, Path(value) if key in ("golden", "book") and value else value)
    args.resume_manifest = manifest


def run_options(args):
    return {key: str(Path(value).resolve()) if isinstance(value, Path) else value
            for key in RUN_OPTIONS for value in [getattr(args, key)]}


def shown(path):
    """A path relative to the repo when it's inside it, for messages and manifests."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def fingerprint(paths):
    """sha256 over files' paths and bytes, to tell whether they changed between starting and resuming a run."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def app_fingerprint():
    return fingerprint([*REPO_ROOT.glob("core/*.py"), *REPO_ROOT.glob("prompts/*.py")])


def saved_ids(path):
    """Ids of the questions already written to a results CSV."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and reader.fieldnames != CSV_COLUMNS:
            sys.exit(f"{path.name} has different columns from this version of run_eval.py; can't append to it")
        return {row["id"] for row in reader}


def append_row(row, path):
    """Add one question's row to a results CSV, writing the header first if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, restval="", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def normalize(text):
    return " ".join(text.split())


def load_chunks(book_path, keep_gutenberg_boilerplate=False, chunk_size=None, chunk_overlap=None):
    """Parse and chunk the book exactly as app.py does, unless chunk_size is given.

    keep_gutenberg_boilerplate=True reproduces the parser as it was before it began stripping the
    Project Gutenberg header and license (2026-11-11), so --backfill can read runs scored back then.
    chunk_size/chunk_overlap split the same text with FALLBACK_SEPARATORS instead of the app's chunker.
    """
    from core import parser
    from core.chunker import chunk_text

    suffix = book_path.suffix.lower()
    if suffix not in (".epub", ".pdf"):
        sys.exit(f"Unsupported file type: {book_path.suffix}")
    parse = parser.parse_epub if suffix == ".epub" else parser.parse_pdf
    strip = parser.strip_gutenberg_boilerplate
    if keep_gutenberg_boilerplate:
        parser.strip_gutenberg_boilerplate = lambda text: text
    try:
        text = parse(str(book_path))
    finally:
        parser.strip_gutenberg_boilerplate = strip
    if chunk_size is None:
        return chunk_text(text)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=FALLBACK_SEPARATORS, length_function=len)
    return splitter.split_text(text)


def chunking_label(args):
    if args.chunk_size is None:
        return "app chunker (2000/100)"
    return f"{args.chunk_size}/{args.chunk_overlap} with fallback separators"


def chunk_stats(chunks, model):
    """Count, average/max length, and how many chunks run past the embedding model's token limit."""
    limit = model.max_seq_length
    tokens = [len(ids) for ids in model.tokenizer(chunks, add_special_tokens=True)["input_ids"]]
    over = [t - limit for t in tokens if t > limit]
    return {
        "chunks": len(chunks),
        "avg_chars": round(sum(map(len, chunks)) / len(chunks)),
        "max_chars": max(map(len, chunks)),
        "avg_tokens": round(sum(tokens) / len(tokens)),
        "max_tokens": max(tokens),
        "token_limit": limit,
        "over_limit": len(over),
        "max_tokens_cut": max(over, default=0),
    }


def remap_golden(golden, chunks, out, label):
    """Write a copy of the golden set whose chunk_ids are the chunks holding each entry's evidence quotes."""
    import copy

    remapped = copy.deepcopy(golden)
    normalized = [normalize(c) for c in chunks]
    for entry in remapped["entries"]:
        entry["chunk_ids"] = evidence_chunks(entry, normalized)
        missing = [q for q in entry["evidence"] if not any(normalize(q) in c for c in normalized)]
        if missing:
            print(f"WARNING: {entry['id']}: {len(missing)} evidence quote(s) are split across chunks "
                  f"and can't be located: {missing}")
        if entry["answerable"] and not entry["chunk_ids"]:
            print(f"WARNING: {entry['id']}: no gold chunks after remapping")
    meta = remapped["meta"]
    meta["num_chunks"] = len(chunks)
    meta["chunker"] = label
    meta["chunk_ids_history"] = (f"Generated by `run_eval --remap-golden` from eval/golden_set.json for {label}; "
                                 "chunk_ids are the chunks containing each entry's evidence quotes. Regenerate it "
                                 "after editing golden_set.json instead of editing this copy.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(remapped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = [len(e["chunk_ids"]) for e in remapped["entries"] if e["answerable"]]
    print(f"Wrote {out}: {len(chunks)} chunks; answerable entries have "
          f"{min(sizes)}-{max(sizes)} gold chunks (avg {sum(sizes) / len(sizes):.1f})")


class HybridSearch:
    """BM25 keyword ranking fused with the vector ranking by reciprocal rank fusion. An eval experiment.

    Each chunk scores 1/(rrf_k + vector rank) + 1/(rrf_k + BM25 rank), summed over both full rankings;
    chunks that share no word with the question get no BM25 term. Ties keep vector order. BM25 is
    rank_bm25's Okapi variant (k1=1.5, b=0.75) over lowercased, Porter-stemmed words without stopwords.
    """

    def __init__(self, first_stage, chunks, rrf_k=60):
        import re
        import time
        from functools import lru_cache

        from nltk.stem import PorterStemmer
        from rank_bm25 import BM25Okapi

        self._time, self._words = time.perf_counter, re.compile(r"[a-z0-9]+").findall
        self._stem = lru_cache(maxsize=None)(PorterStemmer().stem)
        self.first_stage, self.chunks, self.rrf_k = first_stage, chunks, rrf_k
        start = self._time()
        self.bm25 = BM25Okapi([self.tokenize(c) for c in chunks])  # pre-tokenized, so no worker processes
        self.index_seconds = self._time() - start
        self._order, self.added_seconds = {}, []

    def tokenize(self, text):
        return [self._stem(w) for w in self._words(text.lower()) if w not in STOPWORDS]

    def search(self, query, k=5):
        if query not in self._order:
            vector = [m["chunk_num"] for m in self.first_stage(query, k=len(self.chunks))["metadatas"]]
            start = self._time()
            scores = self.bm25.get_scores(self.tokenize(query))
            keyword = sorted((i for i in range(len(self.chunks)) if scores[i] > 0), key=lambda i: -scores[i])
            fused = {cid: 1 / (self.rrf_k + rank) for rank, cid in enumerate(vector, start=1)}
            for rank, cid in enumerate(keyword, start=1):
                fused[cid] += 1 / (self.rrf_k + rank)
            position = {cid: rank for rank, cid in enumerate(vector)}
            self._order[query] = sorted(fused, key=lambda cid: (-fused[cid], position[cid]))
            self.added_seconds.append(self._time() - start)
        top = self._order[query][:k]
        return {"documents": [self.chunks[c] for c in top], "metadatas": [{"chunk_num": c} for c in top],
                "ids": [str(c) for c in top]}

    def latency(self):
        ms = sorted(1000 * t for t in self.added_seconds)
        return {"added_ms": {"mean": round(sum(ms) / len(ms), 1), "median": round(ms[len(ms) // 2], 1),
                             "max": round(ms[-1], 1)},
                "bm25_index_ms": round(1000 * self.index_seconds), "rrf_k": self.rrf_k}


class Reranker:
    """Re-ranks the first-stage top candidates with a cross-encoder. An eval experiment; the app doesn't do this.

    search(query, k) returns the re-ranked candidates first, then the rest of the first-stage ranking in its
    original order, so a large k (used for exact_gold_rank) shows where the full pipeline would put a chunk.
    """

    def __init__(self, first_stage, model_name, candidates):
        import time

        from sentence_transformers import CrossEncoder

        self._time = time.perf_counter
        self.first_stage, self.candidates = first_stage, candidates
        self.model = CrossEncoder(model_name)
        self.model.predict([("warm-up", "warm-up")])  # first call pays one-off setup; keep it out of the timings
        self._order, self.rerank_seconds, self.first_stage_seconds = {}, [], []

    def search(self, query, k=5):
        start = self._time()
        first = self.first_stage(query, k=max(k, self.candidates))
        n = min(self.candidates, len(first["ids"]))
        if query not in self._order:
            self.first_stage_seconds.append(self._time() - start)
            start = self._time()
            scores = self.model.predict([(query, doc) for doc in first["documents"][:n]])
            self.rerank_seconds.append(self._time() - start)
            self._order[query] = sorted(range(n), key=lambda i: -scores[i])  # stable: ties keep first-stage order
        order = (self._order[query] + list(range(n, len(first["ids"]))))[:k]
        return {key: [first[key][i] for i in order] for key in ("documents", "metadatas", "ids")}

    def latency(self):
        def ms(values):
            values = sorted(values)
            return {"mean": round(1000 * sum(values) / len(values)), "median": round(1000 * values[len(values) // 2]),
                    "max": round(1000 * values[-1])}
        return {"rerank_ms": ms(self.rerank_seconds), "first_stage_ms": ms(self.first_stage_seconds),
                "device": str(self.model.device), "pairs_per_question": self.candidates}


def evidence_chunks(entry, normalized_chunks):
    """Ids of the chunks that contain at least one of the entry's evidence quotes."""
    quotes = [normalize(q) for q in entry["evidence"]]
    return sorted({i for q in quotes for i, c in enumerate(normalized_chunks) if q in c})


class ExactSearch:
    """Stands in for the app's Chroma collection during eval runs and ranks every chunk exactly.

    Chroma's HNSW index is approximate and is built a little differently in each process, so the
    same question could retrieve slightly different chunks from one run to the next. With a few
    hundred chunks a brute-force search takes milliseconds and always gives the same ranking.
    VectorStore.search runs unchanged, including how it embeds the question; only the
    nearest-neighbour lookup underneath it is replaced, and only inside the eval.
    """

    def __init__(self, collection):
        import numpy as np

        self._np = np
        self._collection = collection
        config = getattr(collection, "configuration", None) or {}
        space = (config.get("hnsw") or {}).get("space") or (collection.metadata or {}).get("hnsw:space", "l2")
        if space != "l2":
            sys.exit(f"ExactSearch only implements Chroma's 'l2' distance; this collection uses {space!r}. "
                     "Use --approximate-search.")
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        order = sorted(range(len(data["ids"])), key=lambda i: int(data["ids"][i]))
        self._ids = [data["ids"][i] for i in order]
        self._documents = [data["documents"][i] for i in order]
        self._metadatas = [data["metadatas"][i] for i in order]
        self._embeddings = np.asarray(data["embeddings"], dtype=np.float64)[order]

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def query(self, query_embeddings, n_results=10, **kwargs):
        if kwargs:
            raise TypeError(f"ExactSearch.query doesn't support {sorted(kwargs)}")
        result = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for q in self._np.asarray(query_embeddings, dtype=self._np.float64):
            distances = ((self._embeddings - q) ** 2).sum(axis=1)  # squared L2, like Chroma's "l2" space
            top = self._np.argsort(distances, kind="stable")[:n_results]
            result["ids"].append([self._ids[i] for i in top])
            result["documents"].append([self._documents[i] for i in top])
            result["metadatas"].append([self._metadatas[i] for i in top])
            result["distances"].append([float(distances[i]) for i in top])
        return result


def check_golden_set(golden, book_path, chunks, gold_from_evidence):
    """Return ({id: gold chunk ids}, {id: chunks holding an evidence quote}, warnings) after checking
    the set still matches the chunks."""
    meta = golden["meta"]
    warnings = []
    if hashlib.sha256(book_path.read_bytes()).hexdigest() != meta.get("book_sha256"):
        warnings.append("the book file differs from the one the golden set was drafted from")
    if len(chunks) != meta.get("num_chunks"):
        warnings.append(f"the book now splits into {len(chunks)} chunks; the golden set expects "
                        f"{meta.get('num_chunks')} (did the parser or chunker settings change?)")

    normalized = [normalize(c) for c in chunks]
    gold, evidence = {}, {}
    for entry in golden["entries"]:
        quotes = [normalize(q) for q in entry["evidence"]]
        found = evidence_chunks(entry, normalized)
        evidence[entry["id"]] = found
        stale = [q for q in quotes
                 if not any(i < len(normalized) and q in normalized[i] for i in entry["chunk_ids"])]
        if stale:
            warnings.append(f"{entry['id']}: {len(stale)} evidence quote(s) are not in chunks "
                            f"{entry['chunk_ids']}; the quotes are now in chunks {found}")
        gold[entry["id"]] = found if gold_from_evidence else entry["chunk_ids"]
        if gold_from_evidence and entry["answerable"] and not found:
            warnings.append(f"{entry['id']}: no evidence quote found in any chunk; it has no gold chunks")
    return gold, evidence, warnings


def run_pipeline(agent, question):
    """Call the app's Agent.run unchanged, recording the chunks its retrieve step handed to the model."""
    seen = {}
    real_retrieve = agent.retrieve

    def recording_retrieve(q, k=10):
        result = real_retrieve(q, k=k)
        seen.update(result)
        return result

    agent.retrieve = recording_retrieve  # instance attribute only; the Agent class is untouched
    try:
        answer = agent.run(question)
    finally:
        del agent.retrieve
    return answer, seen["documents"], [m["chunk_num"] for m in seen["metadatas"]]


def format_context(documents, chunk_ids):
    # Same layout Agent.generate_answer builds, so the judge sees what the model saw.
    return "\n\n".join(f"[Chunk {cid}]: {doc}" for cid, doc in zip(chunk_ids, documents))


def score_faithfulness(row, judge, documents, context_ids):
    claims = judge.faithfulness(row["question"], format_context(documents, context_ids), row["answer"])
    faithfulness = metrics.faithfulness_score(claims)
    row["faithfulness"] = None if faithfulness is None else round(faithfulness, 3)
    row["claims_supported"] = sum(1 for c in claims if c["supported"])
    row["claims_total"] = len(claims)
    row["unsupported_claims"] = " | ".join(c["claim"] for c in claims if not c["supported"])


def add_precision_columns(row, retrieved_ids, gold_ids):
    for k in PRECISION_KS:
        value = metrics.precision_at_k(retrieved_ids, gold_ids, k)
        row[f"precision@{k}"] = None if value is None else round(value, 3)


def add_citation_columns(row, evidence_ids):
    cited = metrics.cited_chunk_ids(row["answer"])
    row["evidence_chunk_ids"] = " ".join(map(str, evidence_ids))
    row["cited_chunk_ids"] = " ".join(map(str, cited))
    accuracy = metrics.citation_accuracy(cited, evidence_ids) if evidence_ids else None
    row["citation_accuracy"] = None if accuracy is None else round(accuracy, 3)


def classify_row(row, gold_ids, context_ids):
    gold_in_context = bool(set(gold_ids) & set(context_ids))
    if row["answerable"]:
        row["gold_in_context"] = gold_in_context
    row["status"], row["failure_reason"] = metrics.classify(
        row["answerable"], gold_in_context, row["faithfulness"],
        relevance=row.get("relevance"), correctness=row.get("correctness"), abstained=row.get("abstained"),
    )
    row["notes"] = ""
    if row["answerable"] and row["status"] == metrics.PASS and not gold_in_context:
        row["notes"] = "passed without a gold chunk in context; the entry's chunk_ids may be incomplete"


def evaluate_entry(entry, gold_ids, evidence_ids, store, n_chunks, agent, judge, depth):
    answerable = entry["answerable"]
    row = {
        "id": entry["id"],
        "answerable": answerable,
        "question": entry["question"],
        "gold_chunk_ids": " ".join(map(str, gold_ids)),
        "needs_review": entry.get("needs_review", True),
    }

    # 1. Retriever only, with the same k the app uses. Chroma's index is approximate, so this
    # top list can differ slightly from an exhaustive ranking; the metrics score what the app sees.
    top = [m["chunk_num"] for m in store.search(entry["question"], k=depth)["metadatas"]]
    row["retrieved_top5"] = " ".join(map(str, top[:5]))
    row["retrieved_ids"] = " ".join(map(str, top))
    rank = None
    if answerable:
        rank = metrics.first_gold_rank(top, gold_ids)
        row["first_gold_rank"] = rank
        for k in RETRIEVAL_KS:
            row[f"hit@{k}"] = metrics.hit_at_k(rank, k)
        row["reciprocal_rank"] = round(metrics.reciprocal_rank(rank, depth), 3)
        add_precision_columns(row, top, gold_ids)
        # Exhaustive ranking, only to show how far down a missed gold chunk sits (e.g. 208).
        everything = [m["chunk_num"] for m in store.search(entry["question"], k=n_chunks)["metadatas"]]
        row["exact_gold_rank"] = metrics.first_gold_rank(everything, gold_ids)

    if agent is None:
        if answerable:
            in_context = metrics.hit_at_k(rank, depth) == 1
            row["gold_in_context"] = in_context
            row["status"] = "retrieved (generation not run)" if in_context else metrics.RETRIEVAL_MISS
            row["failure_reason"] = "" if in_context else f"no gold chunk in the top {depth}"
        return row

    # 2. Full pipeline, then the judge.
    answer, documents, context_ids = run_pipeline(agent, entry["question"])
    row["answer"] = answer
    row["context_chunk_ids"] = " ".join(map(str, context_ids))
    add_citation_columns(row, evidence_ids if answerable else [])
    score_faithfulness(row, judge, documents, context_ids)
    if answerable:
        verdict = judge.answer_quality(entry["question"], entry["reference_answer"], answer)
        row["relevance"] = verdict["relevance"]
        row["correctness"] = verdict["correctness"]
    else:
        verdict = judge.abstention(entry["question"], entry["reference_answer"], answer)
        row["abstained"] = verdict["abstained"]
    row["judge_reasoning"] = verdict["reasoning"]
    classify_row(row, gold_ids, context_ids)
    return row


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(golden, runs, depth, judge_model, warnings, out_paths, search="exact"):
    """Metrics per question group; with repeats, each is the average with (min-max)."""
    groups = summary.groups_for(golden, runs[0])
    table = {name: [summary.summarize(summary.subset(rows, ids), depth) for rows in runs]
             for name, ids in groups.items()}
    generated = any(r.get("answer") for r in runs[0])
    lines = [("questions", "questions", "count")]
    lines += [(f"hit@{k}", f"hit@{k}", "rate") for k in summary.SUMMARY_KS]
    lines += [(f"MRR@{depth}", "mrr", "rate")]
    lines += [(f"precision@{k}", f"precision@{k}", "rate") for k in PRECISION_KS]
    if generated:
        lines += [("passes", "passes", "count"), ("faithfulness", "faithfulness", "rate"),
                  ("citation accuracy", "citation_accuracy", "rate"), ("answers citing nothing", "uncited", "count"),
                  ("relevance (1-5)", "relevance", "rate"), ("incorrect answers", "incorrect", "count"),
                  ("refusal rate", "refusal_rate", "rate")]

    repeats = f", average (min-max) of {len(runs)} runs" if len(runs) > 1 else ""
    print("\n" + "=" * 78)
    print(f"Summary{repeats}; {search} search" + (f"; judge {judge_model}" if generated else ""))
    print(f"{'':<20}" + "".join(f"{name:>19}" for name in table))
    for label, key, kind in lines:
        cells = []
        for summaries in table.values():
            values = summary.spread(summaries, key)
            cells.append(summary.format_spread(values, digits=1 if kind == "count" else 2))
        print(f"{label:<20}" + "".join(f"{c:>19}" for c in cells))

    per_q = summary.per_question(runs)
    failing = [(qid, q) for qid, q in per_q.items() if q["majority"] in (metrics.RETRIEVAL_MISS, metrics.GENERATION_PROBLEM)]
    flipped = [(qid, q) for qid, q in per_q.items() if q["flipped"]]
    if failing:
        print("\nFailures (majority outcome):")
        for qid, q in failing:
            print(f"  {qid}  {q['majority']:<18}  {q['reasons'][0] if q['reasons'] else ''}")
    if len(runs) > 1:
        print("\nPass/fail changed between repeats: " + (", ".join(
            f"{qid} ({' / '.join(q['statuses'])})" for qid, q in flipped) or "none"))

    print("\nResults: " + ", ".join(str(shown(p)) for p in out_paths))
    if warnings:
        print(f"{len(warnings)} golden-set warning(s) printed above; retrieval scores can't be trusted until they're fixed.")
    unreviewed = sum(1 for r in runs[0] if r["needs_review"])
    if unreviewed:
        print(f"{unreviewed} of {len(runs[0])} golden entries are still marked needs_review.")


def backfill(paths, golden, book_path):
    """Add the precision@k and citation columns to saved results CSVs, in place. No API calls.

    Citation accuracy needs the text of the chunks each run saw. Runs from before the Gutenberg
    cleanup used other chunk numbers, so each file's gold_chunk_ids are checked against the evidence
    quotes to find the chunking it was scored with. Existing columns are written back unchanged.
    """
    entries = {e["id"]: e for e in golden["entries"]}
    chunkings = {}

    def chunking(name):
        if name not in chunkings:
            chunks = load_chunks(book_path, keep_gutenberg_boilerplate=(name != "current parser"))
            chunkings[name] = [normalize(c) for c in chunks]
        return chunkings[name]

    def consistent(rows, normalized):
        # Every gold chunk the run was scored against must hold one of the entry's evidence quotes.
        # (Checked this way round so quotes added to the golden set later, e.g. in its review, don't count
        # against a run that predates them.)
        for row in rows:
            entry = entries.get(row["id"])
            if entry is None:
                return False
            quotes = [normalize(q) for q in entry["evidence"]]
            for g in (int(x) for x in row["gold_chunk_ids"].split()):
                if g >= len(normalized) or not any(q in normalized[g] for q in quotes):
                    return False
        return True

    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        unknown = set(fields) - set(CSV_COLUMNS)
        if unknown:
            sys.exit(f"{path.name}: unexpected columns {sorted(unknown)}; not rewriting it")
        name = next((n for n in ("current parser", "parser before the Gutenberg cleanup")
                     if consistent(rows, chunking(n))), None)
        if name is None:
            print(f"SKIPPED {path.name}: its gold chunk ids don't match the evidence quotes under either parser")
            continue
        normalized = chunking(name)
        for row in rows:
            answerable = row["answerable"] == "True"
            retrieved = (row.get("retrieved_ids") or row.get("context_chunk_ids") or row["retrieved_top5"]).split()
            if answerable:
                add_precision_columns(row, [int(x) for x in retrieved], [int(x) for x in row["gold_chunk_ids"].split()])
            if row.get("answer"):
                add_citation_columns(row, evidence_chunks(entries[row["id"]], normalized) if answerable else [])
        write_csv(rows, path)
        saved = "top 15" if len(retrieved) >= summary.DEPTH else "top 5 only, so no precision@15"
        print(f"{path.name}: {len(rows)} rows, {name} ({len(normalized)} chunks), {saved}")


def rescore(args, golden, gold, chunks, judge, store):
    """Re-judge faithfulness for a saved run's answers, reusing the context the app saw."""
    rows = summary.load_rows(args.rescore)
    print(f"Re-scoring {len(rows)} saved answers from {args.rescore.name} with the current faithfulness judge")
    changes = []
    for n, row in enumerate(rows, start=1):
        if row.get("context_chunk_ids"):
            context_ids = [int(x) for x in row["context_chunk_ids"].split()]
        else:
            # Older CSVs didn't save the context; retrieval is deterministic, so rebuild it.
            context_ids = [m["chunk_num"] for m in store.search(row["question"], k=args.depth)["metadatas"]]
            if " ".join(map(str, context_ids[:5])) != row["retrieved_top5"]:
                print(f"WARNING: {row['id']}: rebuilt context doesn't match the saved top 5; skipping")
                continue
        before = (row["faithfulness"], str(row["claims_total"]), row["status"])
        print(f"[{n}/{len(rows)}] {row['id']}")
        score_faithfulness(row, judge, [chunks[i] for i in context_ids], context_ids)
        row["context_chunk_ids"] = " ".join(map(str, context_ids))
        classify_row(row, gold[row["id"]], context_ids)
        after = (row["faithfulness"], str(row["claims_total"]), row["status"])
        if before != after:
            changes.append((row["id"], before, after, row["unsupported_claims"]))

    out = args.rescore.resolve().with_name(args.rescore.stem + "_rescored.csv")
    write_csv(rows, out)
    fmt = lambda f, n: "-" if f is None else f"{f:.2f} ({n} claims)"
    print(f"\n{len(changes)} of {len(rows)} verdicts changed:")
    for qid, (f0, n0, s0), (f1, n1, s1), unsupported in changes:
        print(f"  {qid}  faithfulness {fmt(f0, n0)} -> {fmt(f1, n1)}   status {s0} -> {s1}")
        if unsupported and s0 != s1:
            print(f"        unsupported: {unsupported}")
    print(f"\nRe-scored results: {shown(out)}")


def main():
    args = parse_args()
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    book_path = args.book or REPO_ROOT / golden["meta"]["book"]
    if not book_path.exists():
        sys.exit(f"Book not found: {book_path}\nsample_books/ is git-ignored, so put the book there or pass --book.")

    if args.backfill:
        backfill(args.backfill, golden, book_path)
        return

    print(f"Parsing and chunking {book_path.name} ({chunking_label(args)}) ...")
    chunks = load_chunks(book_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.remap_golden:
        if args.remap_golden.resolve() == args.golden.resolve():
            sys.exit("--remap-golden would overwrite the golden set it reads; choose another path")
        remap_golden(golden, chunks, args.remap_golden, chunking_label(args))
        return

    if args.show_chunks:
        for cid in (int(x) for x in args.show_chunks.split(",")):
            print(f"\n--- chunk {cid} ({len(chunks[cid])} chars) ---\n{chunks[cid]}")
        return

    entries = golden["entries"]
    if args.ids:
        wanted = set(args.ids.split(","))
        entries = [e for e in entries if e["id"] in wanted]
        if not entries:
            sys.exit(f"No entries match --ids {args.ids}")

    gold, evidence, warnings = check_golden_set(golden, book_path, chunks, args.gold_from_evidence)
    for w in warnings:
        print(f"WARNING: {w}")

    from core.vectorstore import VectorStore
    print(f"Indexing {len(chunks)} chunks with the app's VectorStore ...")
    store = VectorStore()
    store.build_index(chunks)
    stats = chunk_stats(chunks, store.model)
    print(f"{stats['chunks']} chunks, avg {stats['avg_chars']} chars (max {stats['max_chars']}); "
          f"{stats['over_limit']} longer than the embedding model's {stats['token_limit']}-token limit")
    search = "approximate (Chroma HNSW)" if args.approximate_search else "exact"
    if not args.approximate_search:
        store.collection = ExactSearch(store.collection)
    reranker = None
    if args.rerank:
        print(f"Loading re-ranker {args.reranker_model} ...")
        reranker = Reranker(store.search, args.reranker_model, args.candidates)
        store.search = reranker.search  # instance attribute: evaluate_entry and the Agent both go through it
    hybrid = None
    if args.hybrid:
        hybrid = HybridSearch(store.search, chunks, args.rrf_k)
        store.search = hybrid.search

    agent = judge = None
    if not args.retrieval_only:
        from core.agent import Agent  # also loads .env
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set (add it to .env), or use --retrieval-only.")
        agent = Agent(store)
        judge = Judge(agent.client, args.judge_model)

    if args.rescore:
        if judge is None:
            sys.exit("--rescore needs the judge, so it can't be combined with --retrieval-only.")
        rescore(args, golden, gold, chunks, judge, store)
        return

    golden_sha = hashlib.sha256(args.golden.read_bytes()).hexdigest()
    app_sha = app_fingerprint()
    if args.resume:
        manifest_path, manifest = args.resume, args.resume_manifest
        if manifest["golden_sha256"] != golden_sha:
            sys.exit("The golden set has changed since this run started; start a new run instead of resuming.")
        if manifest["app_sha256"] != app_sha:
            sys.exit("The app code (core/, prompts/) has changed since this run started; start a new run instead.")
        if [e["id"] for e in entries] != manifest["question_ids"]:
            sys.exit("The run's questions no longer match the golden set; start a new run instead.")
        if args.approximate_search:
            print("WARNING: resuming rebuilds Chroma's approximate index, so retrieval for the remaining "
                  "questions may differ slightly from the first part of the run.")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "retrieval" if args.retrieval_only else "results"
        manifest_path = RESULTS_DIR / f"{prefix}_{stamp}.json"
        csv_names = [f"{prefix}_{stamp}{f'_r{rep}' if args.repeats > 1 else ''}.csv"
                     for rep in range(1, args.repeats + 1)]
        manifest = {
            "label": args.label,
            "created": datetime.now().isoformat(timespec="seconds"),
            "complete": False,
            "retrieval_only": args.retrieval_only,
            "repeats": args.repeats,
            "questions": len(entries),
            "options": run_options(args),
            "question_ids": [e["id"] for e in entries],
            "golden_sha256": golden_sha,
            "app_sha256": app_sha,
            "csv_files": csv_names,
        }

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest()

    out_paths = [RESULTS_DIR / name for name in manifest["csv_files"]]
    try:
        for rep, out_path in enumerate(out_paths, start=1):
            done = saved_ids(out_path)
            if args.repeats > 1:
                print(f"\n=== Repeat {rep} of {args.repeats} ===" + (f" ({len(done)} already saved)" if done else ""))
            for n, entry in enumerate(entries, start=1):
                if entry["id"] in done:
                    continue
                print(f"[{n}/{len(entries)}] {entry['id']}  {entry['question']}")
                row = evaluate_entry(entry, gold[entry["id"]], evidence[entry["id"]], store, len(chunks),
                                     agent, judge, args.depth)
                append_row(row, out_path)
    except BaseException:
        saved = sum(len(saved_ids(p)) for p in out_paths)
        print(f"\nStopped with {saved} of {len(entries) * len(out_paths)} question results saved. Finish this run "
              f"with:\n    python -m eval.run_eval --resume {shown(manifest_path)}\n")
        raise
    runs = [summary.load_rows(p) for p in out_paths]

    manifest.update({
        "complete": True,
        "completed": datetime.now().isoformat(timespec="seconds"),
        "retrieval_only": args.retrieval_only,
        "judge_model": None if args.retrieval_only else args.judge_model,
        "repeats": args.repeats,
        "questions": len(entries),
        "search": search,
        "chunking": chunking_label(args),
        "chunk_stats": stats,
        "reranker": f"{args.reranker_model}, top {args.candidates} -> {args.depth}" if reranker else None,
        "rerank_latency": reranker.latency() if reranker and reranker.rerank_seconds else None,
        "hybrid": f"BM25 (rank_bm25 Okapi) + vectors, reciprocal rank fusion k={args.rrf_k}" if hybrid else None,
        "hybrid_latency": hybrid.latency() if hybrid and hybrid.added_seconds else None,
        "golden_set": str(shown(args.golden)),
        "golden_review": golden["meta"].get("review"),
    })
    write_manifest()
    print_summary(golden, runs, args.depth, args.judge_model, warnings, out_paths, search)
    if reranker and reranker.rerank_seconds:
        lat = reranker.latency()
        print(f"Re-ranking {args.candidates} chunks per question on {lat['device']}: {lat['rerank_ms']['mean']} ms "
              f"mean, {lat['rerank_ms']['median']} ms median, {lat['rerank_ms']['max']} ms max "
              f"(first-stage search {lat['first_stage_ms']['mean']} ms mean)")
    if hybrid and hybrid.added_seconds:
        lat = hybrid.latency()
        print(f"Hybrid: BM25 index built in {lat['bm25_index_ms']} ms; BM25 + fusion add {lat['added_ms']['mean']} ms "
              f"per question (median {lat['added_ms']['median']}, max {lat['added_ms']['max']})")


if __name__ == "__main__":
    main()
