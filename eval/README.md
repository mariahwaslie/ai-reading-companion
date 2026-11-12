# RAG evaluation

A small evaluation harness for the reading companion's RAG pipeline. It runs the app's own parser, chunker, `VectorStore` and `Agent` unchanged against a hand-checkable golden set, and reports:

- **Retrieval**: hit rate (recall@k) at k = 1, 3, 5, plus hit@15 and MRR@15. The app passes 15 chunks to the model, so hit@15 tells you whether the answer reached the model at all. **Precision@5** and **precision@15** give the share of the top k chunks that are gold chunks.
- **Generation**: an LLM judge scores each answer for **faithfulness** (share of the answer's claims supported by the retrieved chunks), **answer relevance** (1–5), and **correctness** against the reference answer.
- **Citations**: **citation accuracy** is the share of the chunks an answer cites, such as "(Chunk 47)", that contain one of the question's evidence quotes. It is computed from the text, with no model call.
- **Unanswerable questions**: whether the app says it doesn't know instead of inventing an answer.
- **Failure type**: each failed question is flagged as a **retrieval miss** (no gold chunk reached the model) or a **generation problem** (a gold chunk was in the context, but the answer was still wrong, unsupported or off-topic; or the app answered an unanswerable question).

## Files

| File | What it is |
| --- | --- |
| `golden_set.json` | 35 questions about *The Confessions of St. Augustine* (`sample_books/pg3296-images-3.epub`): the original 20 (q01–q20) and 15 harder ones (q21–q35), 6 of them unanswerable |
| `run_eval.py` | The runner: builds the index, runs retrieval, the full pipeline and the judge, writes CSVs and prints a summary |
| `judge.py` | The LLM-as-judge prompts |
| `metrics.py` | Hit rate, MRR, faithfulness score, pass/fail thresholds and failure classification |
| `summary.py` | Per-run and across-repeat summaries, shared by the runner and the report |
| `report.py` | Builds `results/report.html` from the saved runs |
| `review_sheet.py` | Builds `results/review_sheet.html`, a card per golden-set entry for reviewing it by hand |
| `test_metrics.py` | Unit tests for `metrics.py` |
| `results/` | Run outputs and the report (git-ignored) |

## Running it

From the repo root, with the virtual environment active and `OPENAI_API_KEY` in `.env`:

```bash
# Retrieval only: free, no API calls (about a minute, mostly embedding the book)
python -m eval.run_eval --retrieval-only

# Full run: retrieval + the app's answer + judge
python -m eval.run_eval --label "what changed"

# Repeat the full run to see run-to-run noise (averages with min-max)
python -m eval.run_eval --repeats 3 --label "what changed"

# A few questions only
python -m eval.run_eval --ids q09,q13,q18

# Finish a run that stopped partway (e.g. the API quota ran out); saved questions aren't repeated
python -m eval.run_eval --resume eval/results/results_<timestamp>.json

# Re-judge a saved run's answers after changing judge.py, without new answers
python -m eval.run_eval --rescore eval/results/results_<timestamp>.csv

# Add columns from a newer version of the eval (precision@k, citations) to saved CSVs; no API calls
python -m eval.run_eval --backfill eval/results/results_<timestamp>.csv

# Use the app's approximate Chroma index instead of the eval's exact search (see Caveats)
python -m eval.run_eval --retrieval-only --approximate-search

# Unit tests for the metric functions
python -m unittest eval.test_metrics
```

The book lives in `sample_books/`, which is git-ignored, so a fresh clone needs the EPUB copied there (or pass `--book path/to/file.epub`).

A full run makes three model calls per question: one `gpt-4.1-mini` call (the app's own answer) and two judge calls. That's 105 calls for the 35 questions, or 315 with `--repeats 3`. The judge defaults to `gpt-4.1`, a stronger model than the one being graded, so the generator isn't grading itself; change it with `--judge-model`. The faithfulness judge reads the same ~15 chunks the app sent, so each question costs roughly 15k input tokens.

## Report

```bash
python -m eval.report
```

This writes `eval/results/report.html`, a single file with the charts embedded, which opens in any browser without a server. It covers the latest full run: summary cards, hit@k for k = 1–15, each question's gold-chunk rank and faithfulness, a comparison of every saved run, and a table of every question with why each failure failed. A banner appears while any golden-set entry is still marked `needs_review`.

The report needs matplotlib but none of the app's dependencies, so any Python with matplotlib can run it. The project venv doesn't include matplotlib; on this Mac, Homebrew's Python 3.11 has it:

```bash
/opt/homebrew/bin/python3.11 -m eval.report
```

The run comparison uses every complete `results_*.json` manifest in `results/`, labelled with the run's `--label`.

To take the retrieval numbers (hit@k, MRR, precision, gold-chunk ranks) from a different run than the answers, for example a retrieval-only run against a newer golden set, pass both:

```bash
/opt/homebrew/bin/python3.11 -m eval.report --run eval/results/results_<full run>.json --retrieval-run eval/results/retrieval_<run>.json
```

The report then says which numbers come from which run. If the full run was scored before the golden-set review, it also re-checks that run's outcomes against the reviewed gold chunks and reports any that would change.

## Retrieval experiments

Eval-only options for trying retrieval changes before touching the app. They work with `--retrieval-only` (no API calls) or a full run.

```bash
# Another chunk size: split with fallback separators ["\n\n", "\n", ". ", " ", ""], remap the golden set into a copy, then run on the copy
python -m eval.run_eval --chunk-size 1000 --chunk-overlap 150 --remap-golden eval/results/golden_set_1000_150.json
python -m eval.run_eval --retrieval-only --chunk-size 1000 --chunk-overlap 150 --golden eval/results/golden_set_1000_150.json

# Re-rank the top 50 with a cross-encoder and keep the top 15 (downloads BAAI/bge-reranker-base, 1.13 GB, on first use)
python -m eval.run_eval --retrieval-only --rerank

# Hybrid: fuse BM25 keyword ranking with the vector ranking (reciprocal rank fusion, k = 60); needs `pip install rank-bm25`
python -m eval.run_eval --retrieval-only --hybrid
```

- **The remapped copy:** its `chunk_ids` are the chunks that contain each entry's evidence quotes. Regenerate it after editing `golden_set.json`; don't edit the copy.
- **What each run prints:** the chunk count, the average and maximum chunk length, and how many chunks exceed the embedding model's 512-token limit. With `--rerank` or `--hybrid` it also prints the added time per question. The manifest saves both.
- **`exact_gold_rank` with `--rerank`:** this is the rank the whole pipeline would give the chunk. Re-ranked candidates come first, then the rest in embedding order, so a gold chunk outside the candidates keeps its embedding rank. `--candidates` sets how many chunks are re-ranked.

## Output

Each run writes `eval/results/results_<timestamp>.csv` (`_r1`, `_r2`, ... with `--repeats`), plus a `results_<timestamp>.json` manifest.

- **The manifest:** it records the label, the run's settings, the file list, and fingerprints of the golden set and app code.
- **Rows are saved as they're scored:** each question's row is added to its CSV as soon as it's scored, and the manifest is marked complete at the end.
- **`--resume`:** it continues an incomplete run with the same settings. It refuses to resume if the golden set or the app code (`core/`, `prompts/`) changed in between.
- **Incomplete runs:** the report and review sheet ignore them. Each CSV has one row per question:

| Column | Meaning |
| --- | --- |
| `gold_chunk_ids`, `retrieved_top5` | What should have been retrieved, and what the app retrieved |
| `retrieved_ids` | The full top 15 the retriever returned |
| `first_gold_rank` | Rank of the first gold chunk in the app's top 15 (blank if it isn't there) |
| `exact_gold_rank` | Rank in a search over every chunk, to show how far down a miss is (e.g. 208) |
| `hit@1`, `hit@3`, `hit@5`, `reciprocal_rank`, `precision@5`, `precision@15` | Retrieval metrics (answerable questions only) |
| `gold_in_context`, `context_chunk_ids` | Whether a gold chunk was among the chunks the app sent to the model, and which chunks those were |
| `answer` | The app's answer |
| `evidence_chunk_ids`, `cited_chunk_ids`, `citation_accuracy` | Chunks that contain an evidence quote, chunks the answer cites, and the share of cited chunks that are evidence chunks (blank if the answer cites nothing) |
| `faithfulness`, `claims_supported`, `claims_total`, `unsupported_claims` | Faithfulness judge output. Blank faithfulness means the answer made no factual claims |
| `relevance`, `correctness` | Answer judge output (answerable questions) |
| `abstained` | Whether the app said it didn't know (unanswerable questions) |
| `status`, `failure_reason` | `pass`, `retrieval miss` or `generation problem`, and why |
| `judge_reasoning` | The judge's explanation, for spot-checking it |

The summary breaks results out by the question groups in the golden set's `meta.groups` (original 20, new 15) and the total. With `--repeats`, each number is the average with the (min–max) across repeats, and questions that switch between pass and fail are listed.

A question **passes** when faithfulness is at least 0.8, relevance is at least 3, and correctness isn't `incorrect` (or, for unanswerable questions, when the app abstains and its other claims are supported). The thresholds are constants at the top of `metrics.py`; adjust them once you've read a few runs' answers next to their scores. Precision and citation accuracy are reported but don't affect pass/fail.

How to read the two newer metrics:

- **Precision@k** can't reach 1. Most questions have one to three gold chunks, so even perfect retrieval gives at most about 0.36 at k = 5 and 0.12 at k = 15 on this golden set. The summary and report show this best possible value next to it.
- **Citation accuracy** is strict. A cited chunk that supports the answer but doesn't contain an evidence quote counts as wrong, so an incomplete `chunk_ids`/`evidence` list in the golden set lowers it too. Only answerable questions whose answer cites at least one chunk are averaged; the summary also counts answers that cite nothing.

## Reviewing the golden set

Every entry starts with `"needs_review": true`. The questions and reference answers were drafted from the book text, but a person should confirm them before trusting the numbers.

The easiest way is the review sheet: `python -m eval.review_sheet` (with the project venv) writes `eval/results/review_sheet.html`. It has one card per entry, showing:

- the reference answer and evidence quotes;
- each gold chunk with its quotes highlighted;
- the latest full run's answers and judge verdicts;
- chunks worth a look: candidates flagged in the notes, and chunks the answers cited that aren't gold.

Your keep/change choices and notes stay in your browser; "Copy my review" turns them into text.

For each entry:

1. Read its card, or print its chunks: `python -m eval.run_eval --show-chunks 47,74`
2. Check that the reference answer is right and complete, and that `chunk_ids` lists every chunk that answers the question (any one counts as a hit). The `notes` field gives each new question's type and points at borderline chunks.
3. Set `"needs_review": false`.

For the unanswerable entries (`"answerable": false`), confirm the book really doesn't contain the answer.

## When you change chunking

`chunk_ids` are positions in the chunk list, so they shift if the parser, `chunk_size`/`chunk_overlap`, or the book file changes. Each entry also stores short verbatim `evidence` quotes. On every run, the runner checks those quotes against the current chunks and prints a warning naming the affected entries when they've moved.

To compare chunking settings, add `--gold-from-evidence`. Gold chunks are then located from the quotes instead of the stored IDs. A quote that ends up split across two chunks can't be found; the runner warns about that too.

## Caveats

- The app calls the model at its default temperature, so answers (and generation scores) vary between runs. Use `--repeats` before reading much into a small change.
- **Retrieval in the eval is exact; in the app it's approximate.** Chroma's index is approximate, and it is built slightly differently each time, so in the app the same question can retrieve slightly different chunks after each upload. Runs saved before this change show the effect: hit@5 for the original 20 moved between 0.71 and 0.76 depending on the build. To make runs repeatable, `run_eval` keeps the app's `VectorStore` (including how it embeds the question) but replaces the nearest-neighbour lookup underneath it with an exact search over all chunks (`ExactSearch` in `run_eval.py`). With 335 chunks that takes milliseconds, and two separate runs give identical rankings. The app itself is unchanged. The eval therefore measures retrieval as it would be with a perfect index; pass `--approximate-search` to measure the app's index as-is. Runs saved before this change used the approximate index, and each manifest's `search` field records which kind a run used.
- An LLM judge is a second model's opinion. Spot-check `judge_reasoning` and `unsupported_claims` for the failures before acting on them.
- Hit rate counts a question as a hit if any one gold chunk is retrieved, which is lenient for the two-part questions (q21–q25) whose answers span two places in the book.
- 35 questions is enough to catch regressions and compare settings, not to report precise accuracy. One question is about 3 percentage points.
- `eval/results/` is git-ignored, so run outputs stay local.
