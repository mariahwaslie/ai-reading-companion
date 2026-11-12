"""Build a self-contained HTML report from the runs saved in eval/results/.

Run from the repo root with any Python that has matplotlib; the report reads only
the CSVs and the golden set, not the app's dependencies:
    python -m eval.report                                  # latest run -> eval/results/report.html
    python -m eval.report --run eval/results/results_X.json
    python -m eval.report --run eval/results/results_X.json --retrieval-run eval/results/retrieval_Y.json

--retrieval-run takes the retrieval numbers (hit@k, MRR, precision, gold-chunk ranks) from a separate,
usually retrieval-only, run, e.g. one scored against a newer golden set; answer numbers stay from --run.
"""

import argparse
import base64
import html
import io
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval import metrics, summary  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
DEPTH = summary.DEPTH
RANK_CAP = 50

# Recessive chrome, status colors for outcomes, categorical slots (in fixed order) for runs/groups.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUTCOME_COLORS = {
    metrics.PASS: "#0ca30c",
    metrics.RETRIEVAL_MISS: "#ec835a",
    metrics.GENERATION_PROBLEM: "#d03b3b",
}
OUTCOME_TINTS = {
    metrics.PASS: "rgba(12, 163, 12, 0.07)",
    metrics.RETRIEVAL_MISS: "rgba(236, 131, 90, 0.14)",
    metrics.GENERATION_PROBLEM: "rgba(208, 59, 59, 0.11)",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK2,
    "axes.labelcolor": INK2,
})
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PNG_DIR = None  # set by --png-dir to also save each chart as a file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run manifest to report on (default: the latest full run)")
    p.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.json")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "report.html")
    p.add_argument("--retrieval-run", type=Path, metavar="MANIFEST",
                   help="take retrieval metrics and gold-chunk ranks from this run instead of --run")
    p.add_argument("--png-dir", type=Path, help="also write each chart as a PNG here")
    return p.parse_args()


# ---------- data ----------

def run_label(manifest):
    label = manifest.get("label") or manifest["created"]
    repeats = manifest.get("repeats", 1)
    return f"{label} (avg of {repeats})" if repeats > 1 else label


def load_runs():
    """Every full (not retrieval-only) run with a manifest, oldest first."""
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "csv_files" not in manifest or manifest.get("retrieval_only") or not manifest.get("complete", True):
            continue  # retrieval-only, or an interrupted run that hasn't been resumed yet
        _, reps = summary.load_run(path)
        runs.append({"path": path, "manifest": manifest, "reps": reps, "label": run_label(manifest)})
    return sorted(runs, key=lambda r: r["manifest"]["created"])


def tally(values):
    """'correct x3' or 'correct x2, partial x1'."""
    counts = Counter(v for v in values if v)
    return ", ".join(f"{v} ×{n}" for v, n in counts.most_common()) or "-"


def display_rank(row):
    """(value to plot, label for a miss or None). Old CSVs have no exact rank for misses."""
    if row["first_gold_rank"] is not None:
        return row["first_gold_rank"], None
    exact = row.get("exact_gold_rank")
    if exact is None:
        return DEPTH + 1, "> 15"
    if exact > DEPTH:
        return exact, f"rank {exact}"
    # An exhaustive search ranks it inside 15, but the app's approximate top-15 query missed it.
    return DEPTH + 1, f"missed by the approximate search (exact rank {exact})"


def outcome_changes(reps, golden):
    """[(id, repeat, old, new)] for answers whose outcome changes when re-checked against the current gold chunks."""
    gold = {e["id"]: e["chunk_ids"] for e in golden["entries"]}
    changes = []
    for n, rows in enumerate(reps, start=1):
        for r in rows:
            context = summary._ids(r.get("context_chunk_ids"))
            if not context or r["id"] not in gold:
                continue
            status, _ = metrics.classify(r["answerable"], bool(set(gold[r["id"]]) & set(context)), r["faithfulness"],
                                         relevance=r.get("relevance"), correctness=r.get("correctness") or None,
                                         abstained=r.get("abstained"))
            if status != r.get("status"):
                changes.append((r["id"], n, r.get("status"), status))
    return changes


# ---------- charts ----------

def style_axes(ax, grid_axis):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=AXIS, labelcolor=INK2, length=0, labelsize=9, pad=6)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def to_img(fig, name, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    if PNG_DIR:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        (PNG_DIR / f"{name}.png").write_bytes(buf.getvalue())
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def outcome_legend(ax, outcomes, **kwargs):
    handles = [Patch(color=OUTCOME_COLORS[o], label=o) for o in OUTCOME_COLORS if o in outcomes]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2, handlelength=1, handleheight=1, **kwargs)


def hit_curve_chart(groups, rows):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    style_axes(ax, "y")
    ks = list(range(1, DEPTH + 1))
    for color, (name, ids) in zip(SERIES, groups.items()):
        curve = summary.hit_curve(summary.subset(rows, ids))
        if not curve:
            continue
        ax.plot(ks, curve, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=name)
        ax.plot([ks[-1]], [curve[-1]], "o", color=color, ms=7, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{name}  {curve[-1]:.2f}", (ks[-1], curve[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.6, DEPTH + 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 3, 5, 10, 15])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylabel("hit rate (gold chunk in top k)")
    if len(ax.get_lines()) > 2:  # one line + its end dot per series; a single series needs no legend
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    return to_img(fig, "hit_at_k", "Line chart of hit rate at k from 1 to 15 for each question group")


def rank_chart(per_q, ids, rank_rows):
    fig, ax = plt.subplots(figsize=(8, 0.27 * len(ids) + 1.1))
    style_axes(ax, "x")
    outcomes = set()
    for i, qid in enumerate(ids):
        q = per_q[qid]
        value, miss_label = display_rank(rank_rows.get(qid, q["row"]))
        outcomes.add(q["majority"])
        ax.barh(i, min(value, RANK_CAP), height=0.56, color=OUTCOME_COLORS.get(q["majority"], MUTED))
        if miss_label:
            ax.text(min(value, RANK_CAP) + 0.8, i, miss_label, va="center", fontsize=8.5, color=INK2)
    ax.axvline(DEPTH + 0.5, color=INK2, lw=1)
    ax.text(DEPTH + 0.9, -0.95, "top 15 go to the model", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{qid} *" if per_q[qid]["flipped"] else qid for qid in ids])
    ax.set_ylim(len(ids) - 0.4, -1.5)
    ax.set_xlim(0, RANK_CAP + 9)
    ax.set_xticks([1, 5, 10, 15, 20, 30, 40, 50])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "30", "40", "50+"])
    ax.set_xlabel("rank of the first gold chunk (lower is better; axis capped at 50)")
    outcome_legend(ax, outcomes, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return to_img(fig, "gold_rank", "Bar chart of each answerable question's gold chunk rank, colored by outcome")


def faithfulness_chart(per_q, ids):
    fig, ax = plt.subplots(figsize=(8, 0.27 * len(ids) + 1.1))
    style_axes(ax, "x")
    outcomes = set()
    for i, qid in enumerate(ids):
        q = per_q[qid]
        values = [v for v in q["faithfulness"] if v is not None]
        outcomes.add(q["majority"])
        if not values:
            ax.text(0.01, i, "no factual claims", va="center", fontsize=8.5, color=MUTED)
            continue
        avg = mean(values)
        color = OUTCOME_COLORS.get(q["majority"], MUTED)
        ax.barh(i, avg, height=0.56, color=color)
        if avg == 0:  # a zero-length bar draws nothing, so mark the outcome with a dot
            ax.plot([0.006], [i], "o", color=color, ms=7, mec=SURFACE, mew=2, clip_on=False, zorder=3)
        if max(values) > min(values):
            ax.plot([min(values), max(values)], [i, i], color=INK, lw=1.2, solid_capstyle="butt")
            ax.plot([min(values), max(values)], [i, i], "|", color=INK, ms=6, mew=1.2)
        if avg < metrics.FAITHFULNESS_MIN:  # label only the ones below the pass line
            ax.text(avg + 0.025, i, f"{avg:.2f}", va="center", fontsize=8.5, color=INK2, zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
    ax.axvline(metrics.FAITHFULNESS_MIN, color=INK2, lw=1)
    ax.text(metrics.FAITHFULNESS_MIN - 0.01, -0.95, f"pass threshold {metrics.FAITHFULNESS_MIN}",
            fontsize=8.5, color=INK2, va="center", ha="right")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{qid} *" if per_q[qid]["flipped"] else qid for qid in ids])
    ax.set_ylim(len(ids) - 0.4, -1.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("faithfulness: share of the answer's claims supported by the retrieved chunks")
    outcome_legend(ax, outcomes, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return to_img(fig, "faithfulness", "Bar chart of each question's faithfulness score, colored by outcome")


COMPARE = [("hit@5", "hit@5"), ("hit@15", "hit@15"), ("MRR@15", "mrr"), ("precision@5", "precision@5"),
           ("precision@15", "precision@15"), ("pass rate", "pass_rate"), ("faithfulness", "faithfulness"),
           ("citation accuracy", "citation_accuracy"), ("refusal rate", "refusal_rate")]


def comparison_chart(runs):
    runs = runs[-len(SERIES):]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style_axes(ax, "y")
    width = min(0.7 / len(runs), 0.16)  # keeps bars at or under ~24px at the page's width
    for j, (run, color) in enumerate(zip(runs, SERIES)):
        offset = (j - (len(runs) - 1) / 2) * (width + 0.02)
        summaries = [summary.summarize(rows) for rows in run["reps"]]
        for x, (_, key) in enumerate(COMPARE):
            spread = summary.spread(summaries, key)
            if spread is None:
                continue
            avg, lo, hi = spread
            ax.bar(x + offset, avg, width=width, color=color, label=run["label"] if x == 0 else None)
            if hi > lo:
                ax.plot([x + offset] * 2, [lo, hi], color=INK, lw=1.2)
    ax.set_xticks(range(len(COMPARE)))
    ax.set_xticklabels([label.replace(" ", "\n").replace("precision@", "precision\n@") for label, _ in COMPARE])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(runs), fontsize=9,
              labelcolor=INK2, handlelength=1, handleheight=1)
    return to_img(fig, "run_comparison", "Grouped bar chart comparing headline metrics across runs")


# ---------- HTML pieces ----------

def esc(value):
    return html.escape(str(value))


def card(label, value, *subs):
    sub_html = "".join(f'<div class="card-sub">{esc(s)}</div>' for s in subs if s)
    return f'<div class="card"><div class="card-label">{esc(label)}</div><div class="card-value">{esc(value)}</div>{sub_html}</div>'


def group_summaries(golden, reps):
    """({group: [summary per repeat]}, name of the all-questions group)."""
    groups = summary.groups_for(golden, reps[0])
    by_group = {name: [summary.summarize(summary.subset(rows, ids)) for rows in reps] for name, ids in groups.items()}
    return by_group, next((n for n in groups if n.startswith("all ")), list(groups)[-1])


def cards_html(golden, reps, retrieval_reps=None):
    by_group, all_name = group_summaries(golden, reps)
    by_group_r, _ = group_summaries(golden, retrieval_reps or reps)
    overall, overall_r = by_group[all_name], by_group_r[all_name]
    n_runs = len(reps)

    def span(key, digits=2, source=None):
        return summary.format_spread(summary.spread(source or overall, key), digits)

    def per_group(key, digits=2, groups=None):
        groups = groups or by_group
        return [f"{name}: {summary.format_spread(summary.spread(s, key), digits)}"
                for name, s in groups.items() if name != all_name and summary.spread(s, key)]

    def r_span(key):
        return span(key, source=overall_r)

    def r_groups(key):
        return per_group(key, groups=by_group_r)

    def value_and_range(key, fmt):
        spread = summary.spread(overall, key)
        if spread is None:
            return "-", ""
        avg, lo, hi = spread
        rng = f"range {fmt(lo)}–{fmt(hi)} across {n_runs} runs" if hi > lo else (
            f"same in all {n_runs} runs" if n_runs > 1 else "")
        return fmt(avg), rng

    two = lambda v: f"{v:.2f}"
    passes_val, passes_rng = value_and_range("passes", lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
    faith_val, faith_rng = value_and_range("faithfulness", two)
    cite_val, cite_rng = value_and_range("citation_accuracy", two)
    refusal_val, refusal_rng = value_and_range("refusal_rate", lambda v: f"{v:.0%}")
    questions = overall[0]["questions"]

    def ceiling(k):
        value = overall_r[0].get(f"precision_ceiling@{k}")
        return f"best possible {value:.2f}" if value is not None else "not saved for this run"

    uncited = summary.spread(overall, "uncited")
    uncited_note = (f"{summary.format_spread(uncited, 1)} answerable answers cite no chunk"
                    if uncited and uncited[2] > 0 else "")
    retrieval = "".join([
        card("hit@5", r_span("hit@5"), "gold chunk in the top 5", *r_groups("hit@5")),
        card("hit@15", r_span("hit@15"), "gold chunk among the 15 sent to the model", *r_groups("hit@15")),
        card("MRR@15", r_span("mrr"), "mean reciprocal rank", *r_groups("mrr")),
        card("Precision@5", r_span("precision@5"), "share of the top 5 that are gold", ceiling(5),
             *r_groups("precision@5")),
        card("Precision@15", r_span("precision@15"), "share of the top 15 that are gold", ceiling(15),
             *r_groups("precision@15")),
    ])
    answers = "".join([
        card("End-to-end passes", f"{passes_val} / {questions}", passes_rng, *per_group("passes", 1)),
        card("Mean faithfulness", faith_val, faith_rng, *per_group("faithfulness")),
        card("Citation accuracy", cite_val, "cited chunks that hold the evidence quote", cite_rng, uncited_note,
             *per_group("citation_accuracy")),
        card("Refusal rate", refusal_val, f"\"I don't know\" on {overall[0].get('unanswerable', 0)} unanswerable questions",
             refusal_rng),
    ])
    return (f'<div class="cards-label">Retrieval</div><div class="cards">{retrieval}</div>'
            f'<div class="cards-label">Answers</div><div class="cards">{answers}</div>')


def comparison_table(runs, golden, current):
    cols = [(run["label"], [summary.summarize(rows) for rows in run["reps"]]) for run in runs]
    groups = summary.groups_for(golden, current["reps"][0])
    for name, ids in groups.items():
        if not name.startswith("all "):
            cols.append((f"latest run, {name} only",
                         [summary.summarize(summary.subset(rows, ids)) for rows in current["reps"]]))
    head = "".join(f"<th>{esc(label)}</th>" for label, _ in cols)
    body = []
    for label, key in COMPARE + [("questions", "questions")]:
        digits = 0 if key == "questions" else 2
        cells = "".join(f"<td>{esc(summary.format_spread(summary.spread(s, key), digits))}</td>" for _, s in cols)
        body.append(f"<tr><th>{esc(label)}</th>{cells}</tr>")
    return f'<div class="scroll"><table class="compare"><tr><th></th>{head}</tr>{"".join(body)}</table></div>'


def failure_note(q, row):
    notes = []
    if row["answerable"] and row["first_gold_rank"] is None:
        label = display_rank(row)[1]
        notes.append("Gold chunk not among the 15 sent to the model" + (f" ({label})." if label != "> 15" else "."))
    reasons = [r.replace("no gold chunk in the retrieved context", "").strip("; ") for r in q["reasons"]]
    for reason in list(dict.fromkeys(r for r in reasons if r))[:2]:
        notes.append(reason[0].upper() + reason[1:] + ".")
    unsupported = [r.get("unsupported_claims") for r in q["rows"] if r.get("status") != metrics.PASS]
    first = next((u.split(" | ")[0] for u in unsupported if u), "")
    if first:
        notes.append(f"Unsupported: “{first[:200]}{'...' if len(first) > 200 else ''}”")
    if not notes:
        notes = [r.get("notes") for r in q["rows"] if r.get("notes")][:1]
    return " ".join(notes)


def citation_cell(q):
    """'2 of 3 cited chunks hold the quote', or each repeat's count when they differ."""
    if not q["row"]["answerable"]:
        return "n/a (unanswerable)"
    counts = []
    for r in q["rows"]:
        cited = r.get("cited_chunk_ids") or []
        hits = round((r.get("citation_accuracy") or 0) * len(cited))
        counts.append((hits, len(cited)))
    if not q["rows"][0].get("evidence_chunk_ids"):
        return "-"
    if len(set(counts)) == 1:
        hits, total = counts[0]
        return "cites no chunk" if total == 0 else f"{hits} of {total} cited chunks hold the quote"
    per_run = ", ".join(f"{hits}/{total}" if total else "none" for hits, total in counts)
    return f"{per_run} cited chunks hold the quote (per repeat)"


def question_table(per_q, golden_entries, rank_rows):
    rows_html = []
    for qid, q in per_q.items():
        row = q["row"]
        majority = q["majority"] or "-"
        n = len(q["statuses"])
        count = q["statuses"].count(majority)
        outcome = f"{majority}" + (f" ({count} of {n})" if count < n else "")
        if row["answerable"]:
            value, miss_label = display_rank(rank_rows.get(qid, row))
            rank = str(value) if not miss_label else (
                "not in top 15" if miss_label == "> 15" else f"not in top 15 ({miss_label})")
            judged = tally(q["correctness"])
        else:
            rank = "n/a (unanswerable)"
            judged = tally(["refused" if a else "answered" for a in q["abstained"] if a is not None])
        faith_values = [v for v in q["faithfulness"] if v is not None]
        faith = summary.format_spread((mean(faith_values), min(faith_values), max(faith_values))) if faith_values \
            else "no claims"
        flipped = "yes: " + " / ".join(q["statuses"]) if q["flipped"] else "no"
        note = failure_note(q, row) if (majority != metrics.PASS or q["flipped"]) else ""
        question = golden_entries.get(qid, {}).get("question", row["question"])
        tint = OUTCOME_TINTS.get(majority, "transparent")
        border = OUTCOME_COLORS.get(majority, AXIS)
        rows_html.append(
            f'<tr style="background:{tint}">'
            f'<td class="id" style="border-left:4px solid {border}">{esc(qid)}</td><td class="q">{esc(question)}</td>'
            f'<td class="outcome"><span class="dot" style="background:{border}"></span>{esc(outcome)}</td>'
            f'<td class="rank">{esc(rank)}</td><td class="num">{esc(faith)}</td><td>{esc(judged)}</td>'
            f'<td class="cite">{esc(citation_cell(q))}</td>'
            f'<td>{esc(flipped)}</td><td class="note">{esc(note)}</td></tr>')
    head = ("<tr><th>ID</th><th>Question</th><th>Outcome</th><th>Gold chunk rank</th><th>Faithfulness</th>"
            "<th>Correctness / refusal</th><th>Citations</th><th>Flipped between repeats</th>"
            "<th>Why it failed</th></tr>")
    return f'<div class="scroll"><table class="questions">{head}{"".join(rows_html)}</table></div>'


CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --line:#e1e0d9; --ring:rgba(11,11,11,0.10); }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 6px; }
.meta, .caption { color: var(--ink2); margin: 0 0 12px; }
.caption { font-size: 13px; max-width: 760px; }
.banner { display: flex; gap: 10px; align-items: flex-start; background: #fff6df; border: 1px solid #fab219;
          border-left-width: 6px; border-radius: 8px; padding: 12px 14px; margin: 20px 0; }
.banner strong { display: block; }
.banner.info { background: #eef5ff; border-color: #2a78d6; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 8px; }
.cards-label { color: var(--ink2); font-size: 13px; font-weight: 600; margin: 20px 0 0; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; }
.card-label { color: var(--ink2); font-size: 13px; }
.card-value { font-size: 30px; font-weight: 600; margin: 2px 0 4px; }
.card-sub { color: var(--muted); font-size: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px; }
figure img { width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; margin-top: 12px; border: 1px solid var(--ring); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--ink2); font-weight: 600; background: var(--surface); }
td.num, td.rank, table.compare td { font-variant-numeric: tabular-nums; }
td.num, table.compare td, td.id { white-space: nowrap; }
td.outcome { min-width: 120px; }
td.rank { min-width: 100px; }
td.cite { min-width: 130px; }
td.q { min-width: 200px; }
td.note { min-width: 300px; color: var(--ink2); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
code { font-size: 12px; background: #f0efec; padding: 1px 4px; border-radius: 4px; }
"""


def build(args):
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    runs = load_runs()
    if not runs:
        sys.exit("No full runs with a manifest in eval/results/. Run `python -m eval.run_eval` first.")
    current = next((r for r in runs if args.run and r["path"].resolve() == args.run.resolve()), runs[-1])
    manifest, reps = current["manifest"], current["reps"]
    retrieval_manifest, retrieval_reps = (summary.load_run(args.retrieval_run) if args.retrieval_run
                                          else (manifest, reps))
    rank_rows = {r["id"]: r for r in retrieval_reps[0]}
    per_q = summary.per_question(reps)
    answerable_ids = [qid for qid, q in per_q.items() if q["row"]["answerable"]]
    unreviewed = sum(1 for e in golden["entries"] if e.get("needs_review"))
    entries = {e["id"]: e for e in golden["entries"]}
    groups = summary.groups_for(golden, reps[0])
    flipped = [qid for qid, q in per_q.items() if q["flipped"]]
    n_runs = len(reps)

    banner = ""
    if unreviewed:
        banner = (f'<div class="banner" role="note"><span aria-hidden="true">⚠️</span><div>'
                  f'<strong>Golden set not yet human-reviewed.</strong>'
                  f'{unreviewed} of {len(golden["entries"])} entries are still marked <code>needs_review</code>. '
                  f'The scores depend on reference answers and gold chunks a person hasn\'t checked yet.</div></div>')
    elif golden["meta"].get("review") and not manifest.get("golden_review") and args.retrieval_run:
        changes = outcome_changes(reps, golden)
        recheck = ("Re-checking its outcomes against the reviewed gold chunks changes none of them." if not changes else
                   "Re-checked against the reviewed gold chunks, these outcomes would change: "
                   + "; ".join(f"{q} repeat {n}: {a} → {b}" for q, n, a, b in changes) + ".")
        gold = {e["id"]: " ".join(map(str, e["chunk_ids"])) for e in golden["entries"] if e["answerable"]}
        cites_current = all(r.get("evidence_chunk_ids") == gold.get(r["id"]) for rows in reps for r in rows
                            if r["answerable"] and r["id"] in gold)
        banner = (f'<div class="banner info" role="note"><span aria-hidden="true">ℹ️</span><div>'
                  f'<strong>Retrieval and answer numbers come from two runs.</strong>'
                  f'Retrieval (hit@k, MRR, precision and gold-chunk ranks) is from “{esc(retrieval_manifest.get("label"))}”, '
                  f'scored against the golden set '
                  f'{esc(retrieval_manifest.get("golden_review") or "draft")} with '
                  f'{esc(retrieval_manifest.get("search") or "approximate")} search. Answers (passes, faithfulness, '
                  f'relevance, correctness, refusal) are from “{esc(current["label"])}”, scored before the golden-set '
                  f'review. Review changes: {esc(golden["meta"].get("review_changes", ""))} '
                  f'{esc(recheck)} Citation accuracy was '
                  f'{"recomputed against the reviewed set" if cites_current else "computed against the draft set"}.'
                  f'</div></div>')
    elif golden["meta"].get("review") and not manifest.get("golden_review"):
        banner = (f'<div class="banner" role="note"><span aria-hidden="true">⚠️</span><div>'
                  f'<strong>This run was scored against the golden set before its review.</strong>'
                  f'The golden set was {esc(golden["meta"]["review"])}; rerun the eval to score against the '
                  f'reviewed set. {esc(golden["meta"].get("review_changes", ""))}</div></div>')

    exact = retrieval_manifest.get("search") == "exact"
    search_note = ("Retrieval used an exact search over every chunk, so it is identical in every run." if exact else
                   "Retrieval used the app's approximate Chroma index, which is rebuilt for every run, so a separate "
                   "run can rank a few chunks differently.")
    if args.retrieval_run:
        repeats_note = (f"Answer numbers are averaged over {n_runs} repeats; ranges show min–max. " if n_runs > 1
                        else "") + search_note
    else:
        repeats_note = (f"Averaged over {n_runs} repeats; ranges show min–max. The repeats share one vector index, "
                        f"so their retrieval numbers are identical. " if n_runs > 1 else "Single run. ") + search_note
    no_p15 = [r["label"] for r in runs if not any(summary.summarize(rows).get("precision@15") is not None
                                                  for rows in r["reps"])]
    def names(labels):
        return ", ".join(f"“{label}”" for label in labels)

    fewer = [r["label"] for r in runs if r["manifest"].get("questions", 0) < manifest.get("questions", 0)]
    review = golden["meta"].get("review")
    draft = [r["label"] for r in runs if review and not r["manifest"].get("golden_review")]
    approx = [r["label"] for r in runs if r["manifest"].get("search") != "exact"]
    history_note = " ".join(filter(None, [
        f"{names(fewer)} covered fewer questions and used the original faithfulness judge, so compare the "
        f"“only” columns in the table for like-for-like numbers on the same questions." if fewer else "",
        f"{names(draft)} were scored against the golden set as drafted, before it was {review}. "
        f"{golden['meta'].get('review_changes', '')}" if draft and len(draft) < len(runs) else "",
        f"{names(approx)} retrieved with the app's approximate Chroma index; later runs use an exact search."
        if approx and len(approx) < len(runs) else "",
    ]))
    p15_note = (f"{' and '.join(no_p15)} saved only the top 5 retrieved chunks, so precision@15 is blank for "
                f"{'them' if len(no_p15) > 1 else 'it'}." if no_p15 else "")
    flips_note = (f"Questions marked * changed between pass and fail across repeats: {', '.join(flipped)}."
                  if flipped else ("No question changed between pass and fail across repeats." if n_runs > 1 else ""))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Evaluation Report</title><style>{CSS}</style></head>
<body><main>
<h1>RAG evaluation report</h1>
<p class="meta">Run: <strong>{esc(current["label"])}</strong> · {esc(manifest["created"].replace("T", " "))}
 · {esc(manifest.get("questions", len(reps[0])))} questions · judge {esc(manifest.get("judge_model") or "-")}
 · book: {esc(golden["meta"].get("book_title", ""))}
 · golden set: {esc(manifest.get("golden_review") or "draft (before review)")}
{f' · retrieval: <strong>{esc(retrieval_manifest.get("label"))}</strong> ({esc(retrieval_manifest["created"].replace("T", " "))})' if args.retrieval_run else ""}</p>
{banner}
<p class="caption">{esc(repeats_note)}</p>
{cards_html(golden, reps, retrieval_reps if args.retrieval_run else None)}
<p class="caption" style="margin-top:12px">Precision@k is the share of the top k chunks that are gold chunks.
Most questions have only one to three gold chunks, so even perfect retrieval stays well below 1; compare it with
the best possible value. Citation accuracy is the share of the chunks an answer cites (for example “(Chunk 47)”)
that contain one of the question's evidence quotes, averaged over answerable questions whose answer cites
something. It is strict: a cited chunk that supports the answer without holding the quote counts as wrong, so
gaps in the golden set's gold chunks lower it too. Neither metric affects pass/fail.</p>

<h2>Hit rate by k</h2>
<p class="caption">Share of answerable questions whose gold chunk is in the top k results of the app's own retriever.
The app sends the top 15 chunks to the model, so k = 15 is the ceiling for a correct, grounded answer.</p>
<figure>{hit_curve_chart(groups, retrieval_reps[0])}</figure>

<h2>Where the gold chunk ranks</h2>
<p class="caption">Rank of the first gold chunk for each answerable question (unanswerable questions have none).
Bars past the line never reach the model. Ranks beyond 50 are capped and labelled with their true rank.
Color shows the question's majority outcome; a green bar past the line passed without its gold chunk, which
usually means another chunk also holds the answer (see the table). {esc(flips_note)}</p>
<figure>{rank_chart(per_q, answerable_ids, rank_rows)}</figure>

<h2>Faithfulness per question</h2>
<p class="caption">Share of each answer's claims that the retrieved chunks support, averaged over repeats; whiskers
show min–max. A question fails below {metrics.FAITHFULNESS_MIN}. "No factual claims" usually means a plain
"I don't know".</p>
<figure>{faithfulness_chart(per_q, list(per_q))}</figure>

<h2>Run comparison</h2>
<p class="caption">Headline metrics for every saved run, oldest first; whiskers show min–max for repeated runs.
{esc(history_note)} {esc(p15_note)}</p>
<figure>{comparison_chart(runs)}</figure>
{comparison_table(runs, golden, current)}

<h2>Every question</h2>
<p class="caption">Rows are colored by majority outcome. Faithfulness shows the average with (min–max) when
repeats disagree.</p>
{question_table(per_q, entries, rank_rows)}

<footer>Generated {esc(datetime.now().strftime("%Y-%m-%d %H:%M"))} by <code>python -m eval.report</code> from
{esc(", ".join(manifest["csv_files"]))}.</footer>
</main></body></html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    return args.out, current


def main():
    global PNG_DIR
    args = parse_args()
    PNG_DIR = args.png_dir
    out, current = build(args)
    print(f"Report for '{current['label']}': {out}")


if __name__ == "__main__":
    main()
