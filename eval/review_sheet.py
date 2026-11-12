"""Build eval/results/review_sheet.html: one card per golden-set entry, for checking the set by hand.

Run from the repo root with the project venv (it chunks the book with the app's own code):
    python -m eval.review_sheet

Each card shows the question, reference answer, evidence quotes, every gold chunk with its quotes
highlighted, the latest full run's answers and judge verdicts, and chunks worth a look (candidates
flagged in the entry's notes, and chunks the answers cited that aren't gold). The keep/change choice
and notes you add are saved in your browser only; "Copy my review" turns them into text to paste back.
"""

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval import metrics, summary  # noqa: E402
from eval.run_eval import load_chunks  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Passages to point at in the candidate chunks flagged for q23 and q29, with a one-line read of each.
# They're suggestions for the review, not decisions.
SUGGESTED = {
    35: (["This book of his contains an exhortation to philosophy, and is called \"Hortensius.\""],
         "Introduces the Hortensius and how it fired his love of wisdom. The line about what was missing "
         "(\"the name of Christ was not in it\") is in chunk 36, which is already gold."),
    134: (["But what mystery there lay in \"The Word was made flesh,\" I could not even imagine."],
          "About his own view of the Incarnation after the Platonists, not what their books left out."),
    135: (["But having then read those books of the Platonists",
           "For where was that charity building upon the foundation of humility, which is Christ Jesus? "
           "or when should these books teach me it?"],
          "Says what the Platonist books couldn't teach him (humility, Christ). A partial answer to the "
          "second half of q23."),
    84: (["And it seemed to me better to believe Thee to have created no evil (which to me ignorant seemed "
          "not some only, but a bodily substance"],
         "His earlier, Manichaean-era view that evil was a bodily substance: the opposite of the answer."),
    117: (["Whence then came I to will evil and nill good"],
          "Asks where evil comes from; doesn't reach the conclusion that it isn't a substance."),
    156: (["two contrary souls, of two contrary substances, from two contrary principles"],
          "Argues against the Manichaean idea of two opposed substances; supports the answer only indirectly."),
}

CHUNK_REF = re.compile(r"\bchunks?\s+(\d+(?:\s*(?:-|–|,\s*and|,|and)\s*\d+)*)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.json")
    p.add_argument("--run", type=Path, help="run manifest to show answers from (default: the latest full run)")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "review_sheet.html")
    return p.parse_args()


def latest_full_run():
    manifests = []
    for path in RESULTS_DIR.glob("results_*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest.get("retrieval_only"):
            manifests.append((manifest["created"], path))
    if not manifests:
        sys.exit("No full run in eval/results/. Run `python -m eval.run_eval` first.")
    return max(manifests)[1]


def note_chunks(note):
    """Chunk ids a note mentions: 'chunk 35', 'chunks 134-135', 'Chunks 84, 117 and 156'."""
    ids = []
    for match in CHUNK_REF.finditer(note or ""):
        for part in re.split(r"\s*(?:,\s*and|,|and)\s*", match.group(1)):
            bounds = [int(x) for x in re.split(r"\s*[-–]\s*", part) if x]
            for cid in range(bounds[0], bounds[-1] + 1):
                if cid not in ids:
                    ids.append(cid)
    return ids


def esc(value):
    return html.escape(str(value))


def highlight(text, marks):
    """Escape text and wrap each phrase in <mark>. marks: [(css class, [phrases])]. Whitespace is flexible."""
    spans = []
    for css, phrases in marks:
        for phrase in phrases:
            pattern = r"\s+".join(re.escape(word) for word in phrase.split())
            spans += [(m.start(), m.end(), css) for m in re.finditer(pattern, text)]
    out, pos = [], 0
    for start, end, css in sorted(spans):
        if start < pos:
            continue  # overlaps a span already marked
        out += [esc(text[pos:start]), f'<mark class="{css}">', esc(text[start:end]), "</mark>"]
        pos = end
    out.append(esc(text[pos:]))
    return "".join(out), len(spans)


def chunk_box(cid, chunks, title, marks, note="", open_=True, expect_quote=False):
    body, found = highlight(chunks[cid], marks)
    missing = ""
    if expect_quote and not found:
        missing = '<p class="warn">None of the quotes was found in this chunk.</p>'
    note_html = f'<p class="read">{esc(note)}</p>' if note else ""
    return (f'<details class="chunk"{" open" if open_ else ""}><summary>{title}</summary>{note_html}{missing}'
            f'<div class="chunk-text">{body}</div></details>')


def outcome_badge(status):
    css = {metrics.PASS: "pass", metrics.RETRIEVAL_MISS: "miss", metrics.GENERATION_PROBLEM: "gen"}.get(status, "")
    return f'<span class="badge {css}">{esc(status or "-")}</span>'


def verdict_table(q):
    head = "<tr><th>Repeat</th><th>Outcome</th><th>Faithfulness</th><th>Judge</th><th>Cited chunks</th></tr>"
    rows = []
    for n, r in enumerate(q["rows"], start=1):
        faith = "no claims" if r["faithfulness"] is None else (
            f'{r["faithfulness"]:.2f} ({r.get("claims_supported", "")}/{r.get("claims_total", "")} claims)')
        if r["answerable"]:
            judge = f'{r.get("correctness") or "-"}, relevance {r.get("relevance") or "-"}/5'
        else:
            judge = "refused (said it doesn't know)" if r.get("abstained") else "answered anyway"
        cited = ", ".join(map(str, r.get("cited_chunk_ids") or [])) or "none"
        reason = f'<div class="why">{esc(r["failure_reason"])}</div>' if r.get("failure_reason") else ""
        rows.append(f"<tr><td>{n}</td><td>{outcome_badge(r.get('status'))}{reason}</td><td>{esc(faith)}</td>"
                    f"<td>{esc(judge)}</td><td>{esc(cited)}</td></tr>")
    return f'<table class="verdicts">{head}{"".join(rows)}</table>'


def retrieval_line(row, gold):
    if not row["answerable"]:
        return ""
    context = summary._ids(row.get("context_chunk_ids"))
    sent = [cid for cid in gold if cid in context]
    if sent:
        ranks = ", ".join(f"chunk {cid} at rank {context.index(cid) + 1}" for cid in sent)
        return f"Sent to the model in the latest run: {ranks} of 15."
    exact = row.get("exact_gold_rank")
    return ("No gold chunk was among the 15 sent to the model in the latest run"
            + (f"; the first one ranks {exact} of all chunks." if exact else "."))


def card(entry, group, q, chunks):
    qid = entry["id"]
    gold = entry["chunk_ids"]
    quotes = entry["evidence"]
    note = entry.get("notes", "")
    kind = note.split(".")[0].replace("Type: ", "") if note.startswith("Type:") else ""
    majority = q["majority"] if q else ""
    tags = [f'<span class="tag">{esc(group)}</span>'] + ([f'<span class="tag">{esc(kind)}</span>'] if kind else [])
    if not entry["answerable"]:
        tags.append('<span class="tag">unanswerable</span>')

    parts = [f'<div class="card-head"><h2>{esc(qid)}</h2>{"".join(tags)}'
             f'<span class="spacer"></span>{outcome_badge(majority) if q else ""}</div>',
             f'<div class="field"><div class="label">Question</div><p class="question">{esc(entry["question"])}</p></div>',
             f'<div class="field"><div class="label">Reference answer</div><p>{esc(entry["reference_answer"])}</p></div>']
    if quotes:
        items = "".join(f"<li><mark class=\"ev\">{esc(x)}</mark></li>" for x in quotes)
        parts.append(f'<div class="field"><div class="label">Evidence quotes</div><ul class="quotes">{items}</ul></div>')
    if note:
        parts.append(f'<div class="field"><div class="label">Notes</div><p class="muted">{esc(note)}</p></div>')

    # Gold chunks
    if gold:
        boxes = "".join(chunk_box(cid, chunks, f"Gold chunk {cid}", [("ev", quotes)], expect_quote=True)
                        for cid in gold)
        line = retrieval_line(q["row"], gold) if q else ""
        parts.append(f'<div class="field"><div class="label">Gold chunks ({len(gold)})</div>'
                     f'{f"<p class=muted>{esc(line)}</p>" if line else ""}{boxes}</div>')
    elif entry["answerable"]:
        parts.append('<div class="field"><div class="label">Gold chunks</div><p class="warn">None listed.</p></div>')
    else:
        parts.append('<div class="field"><div class="label">Gold chunks</div><p class="muted">None: the book '
                     'shouldn\'t contain the answer. Check the chunks the answers cited below.</p></div>')

    # Latest run
    if q:
        r1 = q["rows"][0]
        unsupported = [c for r in q["rows"] for c in (r.get("unsupported_claims") or "").split(" | ") if c]
        unsupported_html = ("<div class=\"label sub\">Unsupported claims (judge)</div><ul class=\"claims\">"
                            + "".join(f"<li>{esc(c)}</li>" for c in dict.fromkeys(unsupported)) + "</ul>"
                            if unsupported else "")
        others = "".join(f'<details class="answer-more"><summary>Answer, repeat {n}</summary>'
                         f'<div class="answer">{esc(r.get("answer") or "")}</div></details>'
                         for n, r in enumerate(q["rows"][1:], start=2))
        parts.append(f'<div class="field"><div class="label">Latest run: answers and judge verdicts</div>'
                     f'{verdict_table(q)}<div class="label sub">Answer, repeat 1</div>'
                     f'<div class="answer">{esc(r1.get("answer") or "")}</div>'
                     f'<div class="label sub">Judge\'s reasoning, repeat 1</div>'
                     f'<p class="muted">{esc(r1.get("judge_reasoning") or "-")}</p>{unsupported_html}{others}</div>')

    # Chunks worth a look
    shown = set(gold)
    flagged = [cid for cid in note_chunks(note) if cid not in shown and cid < len(chunks)]
    candidate_note = bool(re.search(r"should count as (gold|hits)", note))
    if flagged:
        title = "Candidate extra gold chunks (flagged in the notes)" if candidate_note else "Chunks mentioned in the notes"
        boxes = []
        for cid in flagged:
            phrases, read = SUGGESTED.get(cid, ([], ""))
            title_ = f"Candidate chunk {cid}" if candidate_note else f"Chunk {cid}"
            boxes.append(chunk_box(cid, chunks, title_, [("sug", phrases)],
                                   note=f"My read: {read}" if read else "", open_=candidate_note))
        hint = ('<p class="muted">Highlighted in blue: the passage I\'d look at. Add a chunk to chunk_ids if it '
                'answers the question on its own.</p>' if candidate_note and any(c in SUGGESTED for c in flagged) else "")
        parts.append(f'<div class="field"><div class="label">{title}</div>{hint}{"".join(boxes)}</div>')
        shown |= set(flagged)
    if q:
        cited = []
        for r in q["rows"]:
            cited += [cid for cid in (r.get("cited_chunk_ids") or []) if cid not in shown and cid not in cited
                      and cid < len(chunks)]
        if cited:
            boxes = "".join(chunk_box(cid, chunks, f"Chunk {cid}", [("ev", quotes)], open_=False) for cid in cited)
            parts.append(f'<div class="field"><div class="label">Other chunks the answers cited ({len(cited)})</div>'
                         f'<p class="muted">Not gold. If one of them answers the question, it may belong in '
                         f'chunk_ids{"; for an unanswerable entry, the entry may be answerable after all" if not entry["answerable"] else ""}.</p>'
                         f'{boxes}</div>')

    # Review fields
    parts.append(f'''<div class="review">
<div class="field-inline"><span class="label">needs_review</span><code>{json.dumps(entry.get("needs_review", True))}</code></div>
<fieldset><legend class="label">Your call</legend>
<label><input type="radio" name="verdict-{qid}" value="keep" data-q="{qid}"> Keep as is</label>
<label><input type="radio" name="verdict-{qid}" value="change" data-q="{qid}"> Change</label></fieldset>
<label class="notes-label" for="notes-{qid}">Notes or the change you want</label>
<textarea id="notes-{qid}" data-q="{qid}" rows="2"></textarea></div>''')
    return f'<article class="card" id="{esc(qid)}" data-outcome="{esc(majority)}">{"".join(parts)}</article>'


CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --line:#e1e0d9; --ring:rgba(11,11,11,0.10); --ev:#fde68a; --sug:#bfdbfe;
        --pass:#0ca30c; --miss:#ec835a; --gen:#d03b3b; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0; }
p { margin: 0 0 8px; }
.meta, .muted { color: var(--ink2); }
.intro { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; margin: 16px 0; }
.intro ol { margin: 6px 0 0; padding-left: 20px; }
.toolbar { position: sticky; top: 0; z-index: 2; background: var(--page); padding: 10px 0; border-bottom: 1px solid var(--line);
           display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip { font-size: 12px; text-decoration: none; color: var(--ink); border: 1px solid var(--line); border-radius: 999px;
        padding: 2px 8px; background: var(--surface); border-left: 4px solid var(--line); }
.chip[data-outcome="pass"] { border-left-color: var(--pass); }
.chip[data-outcome="retrieval miss"] { border-left-color: var(--miss); }
.chip[data-outcome="generation problem"] { border-left-color: var(--gen); }
.chip.done { background: #e8f5e8; }
.chip.change { background: #fdecea; }
.progress { margin-left: auto; font-size: 13px; color: var(--ink2); }
button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--ink2);
         background: var(--surface); cursor: pointer; }
#export { width: 100%; margin-top: 8px; font: 12px/1.4 ui-monospace, monospace; display: none; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 18px 20px; margin: 20px 0;
        scroll-margin-top: 70px; }
.card-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
.spacer { flex: 1; }
.tag { font-size: 12px; color: var(--ink2); background: #f0efec; border-radius: 999px; padding: 1px 8px; }
.badge { font-size: 12px; font-weight: 600; border-radius: 999px; padding: 1px 9px; border: 1px solid var(--line); white-space: nowrap; }
.badge.pass { color: #075e07; border-color: var(--pass); }
.badge.miss { color: #8a3a17; border-color: var(--miss); }
.badge.gen { color: #8f1f1f; border-color: var(--gen); }
.field { margin-top: 14px; }
.label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--ink2); margin-bottom: 4px; }
.label.sub { margin-top: 10px; }
.question { font-weight: 600; }
.quotes, .claims { margin: 0; padding-left: 20px; }
mark.ev { background: var(--ev); padding: 0 1px; border-radius: 2px; }
mark.sug { background: var(--sug); padding: 0 1px; border-radius: 2px; }
details.chunk, details.answer-more { border: 1px solid var(--line); border-radius: 8px; margin-top: 8px; background: #fff; }
details summary { cursor: pointer; padding: 6px 10px; font-size: 13px; font-weight: 600; color: var(--ink2); }
.chunk-text { white-space: pre-wrap; font-size: 13.5px; line-height: 1.6; max-height: 240px; overflow-y: auto;
              padding: 4px 12px 10px; border-top: 1px solid var(--line); }
.read { font-size: 13px; color: #1e3a8a; background: #eff6ff; margin: 0; padding: 6px 12px; border-top: 1px solid var(--line); }
.warn { color: #8f1f1f; font-size: 13px; padding: 0 12px; }
.answer { white-space: pre-wrap; font-size: 13.5px; background: #fff; border: 1px solid var(--line); border-radius: 8px;
          padding: 8px 12px; max-height: 260px; overflow-y: auto; }
details.answer-more .answer { border: 0; border-top: 1px solid var(--line); border-radius: 0; }
table.verdicts { border-collapse: collapse; width: 100%; font-size: 13px; }
table.verdicts th, table.verdicts td { text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid var(--line); }
table.verdicts th { color: var(--ink2); font-weight: 600; }
.why { font-size: 12px; color: var(--ink2); margin-top: 2px; }
.review { margin-top: 16px; padding-top: 12px; border-top: 1px dashed var(--line); display: grid; gap: 6px; }
.field-inline { display: flex; gap: 8px; align-items: baseline; }
fieldset { border: 0; padding: 0; margin: 0; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
fieldset legend { float: left; margin-right: 8px; }
.notes-label { font-size: 13px; color: var(--ink2); }
textarea { width: 100%; font: inherit; font-size: 14px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; }
code { font-size: 13px; background: #f0efec; padding: 1px 5px; border-radius: 4px; }
@media (max-width: 600px) { .card { padding: 14px; } }
"""

SCRIPT = """
const KEY = "golden-set-review";
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } }
function save(state) { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
const state = load();
function refresh() {
  let done = 0;
  document.querySelectorAll(".card").forEach(card => {
    const s = state[card.id] || {};
    const chip = document.querySelector(`.chip[href="#${card.id}"]`);
    chip.classList.toggle("done", s.verdict === "keep");
    chip.classList.toggle("change", s.verdict === "change");
    if (s.verdict) done++;
  });
  document.getElementById("progress").textContent = `${done} of ${document.querySelectorAll(".card").length} reviewed`;
}
document.querySelectorAll("input[type=radio]").forEach(input => {
  const s = state[input.dataset.q] || {};
  input.checked = s.verdict === input.value;
  input.addEventListener("change", () => {
    state[input.dataset.q] = {...(state[input.dataset.q] || {}), verdict: input.value}; save(state); refresh();
  });
});
document.querySelectorAll("textarea[data-q]").forEach(area => {
  area.value = (state[area.dataset.q] || {}).notes || "";
  area.addEventListener("input", () => {
    state[area.dataset.q] = {...(state[area.dataset.q] || {}), notes: area.value}; save(state);
  });
});
document.getElementById("copy").addEventListener("click", () => {
  const lines = [];
  document.querySelectorAll(".card").forEach(card => {
    const s = state[card.id] || {};
    if (s.verdict || (s.notes || "").trim()) lines.push(`${card.id}: ${s.verdict || "no call"}${s.notes ? " - " + s.notes.trim() : ""}`);
  });
  const text = lines.join("\\n") || "(nothing reviewed yet)";
  const out = document.getElementById("export");
  out.style.display = "block"; out.value = text; out.rows = Math.min(12, lines.length + 1); out.select();
  if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
});
function showMark(details) {
  const box = details.querySelector(".chunk-text"), mark = box && box.querySelector("mark");
  if (mark) box.scrollTop = Math.max(0, mark.offsetTop - box.offsetTop - 40);
}
document.querySelectorAll("details.chunk").forEach(d => {
  if (d.open) showMark(d);
  d.addEventListener("toggle", () => { if (d.open) showMark(d); });
});
refresh();
"""


def build(args):
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    book = REPO_ROOT / golden["meta"]["book"]
    chunks = load_chunks(book)
    if len(chunks) != golden["meta"].get("num_chunks"):
        print(f"WARNING: the book splits into {len(chunks)} chunks; the golden set expects "
              f"{golden['meta'].get('num_chunks')}. Chunk ids may not line up.")
    run_path = args.run or latest_full_run()
    manifest, reps = summary.load_run(run_path)
    per_q = summary.per_question(reps)
    group_of = {qid: name for name, ids in golden["meta"].get("groups", {}).items() for qid in ids}
    entries = golden["entries"]
    unreviewed = sum(1 for e in entries if e.get("needs_review", True))
    label = manifest.get("label") or manifest["created"]

    chips = "".join(f'<a class="chip" href="#{esc(e["id"])}" data-outcome="{esc(per_q.get(e["id"], {}).get("majority", ""))}">'
                    f'{esc(e["id"])}</a>' for e in entries)
    cards = "".join(card(e, group_of.get(e["id"], ""), per_q.get(e["id"]), chunks) for e in entries)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden Set Review</title><style>{CSS}</style></head>
<body><main>
<h1>Golden-set review sheet</h1>
<p class="meta">{len(entries)} entries · {unreviewed} marked <code>needs_review</code> · answers from the run
“{esc(label)}” ({esc(manifest["created"].replace("T", " "))}, {len(reps)} repeat{"s" if len(reps) > 1 else ""})
· book: {esc(golden["meta"].get("book_title", ""))}</p>
<div class="intro"><strong>For each entry, check:</strong><ol>
<li>The reference answer is right and complete.</li>
<li>Every chunk that answers the question on its own is in <code>chunk_ids</code> (any one counts as a retrieval hit).
Evidence quotes are highlighted in yellow; passages I suggest looking at are in blue.</li>
<li>For unanswerable entries, the book really doesn't contain the answer.</li></ol>
<p class="muted" style="margin-top:8px">Your “keep / change” choices and notes are saved in this browser only.
When you're done, use “Copy my review” and paste the text to me. Chip borders show the latest run's outcome
(green pass, orange retrieval miss, red generation problem).</p></div>
<div class="toolbar">{chips}<span class="progress" id="progress"></span><button id="copy" type="button">Copy my review</button>
<textarea id="export" readonly aria-label="Your review as text"></textarea></div>
{cards}
<p class="muted">Generated {esc(datetime.now().strftime("%Y-%m-%d %H:%M"))} by <code>python -m eval.review_sheet</code>
from {esc(args.golden.name)} and {esc(", ".join(manifest["csv_files"]))}.</p>
</main><script>{SCRIPT}</script></body></html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    return args.out


def main():
    out = build(parse_args())
    print(f"Review sheet: {out}")


if __name__ == "__main__":
    main()
