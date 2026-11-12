"""Scoring helpers. Pure functions, no model calls, so they are easy to test."""

import re

# Pass/fail thresholds for generation. Starting points: tune them after you
# have read a few runs' answers next to their scores.
FAITHFULNESS_MIN = 0.8  # share of the answer's claims the retrieved chunks support
RELEVANCE_MIN = 3       # judge score on a 1-5 scale

RETRIEVAL_MISS = "retrieval miss"
GENERATION_PROBLEM = "generation problem"
PASS = "pass"


def first_gold_rank(retrieved_ids, gold_ids):
    """1-based rank of the first gold chunk in the retrieved list, or None if absent."""
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return rank
    return None


def hit_at_k(rank, k):
    """1 if a gold chunk appears in the top k, else 0 (recall@k with 'any gold chunk' as a hit)."""
    return int(rank is not None and rank <= k)


def reciprocal_rank(rank, cutoff=None):
    """1/rank, or 0 when there is no gold chunk within the cutoff (MRR@cutoff)."""
    if rank is None or (cutoff is not None and rank > cutoff):
        return 0.0
    return 1.0 / rank


def precision_at_k(retrieved_ids, gold_ids, k):
    """Share of the top k retrieved chunks that are gold chunks, or None if fewer than k were saved.

    A question with one gold chunk can reach at most 1/k, so compare against precision_ceiling.
    """
    if len(retrieved_ids) < k:
        return None
    gold = set(gold_ids)
    return sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in gold) / k


def precision_ceiling(gold_ids, k):
    """Best possible precision@k: every gold chunk ranked first."""
    return min(len(set(gold_ids)), k) / k


# "(Chunk 12)", "(Chunks 4, 5 and 7)", "[Citations: Chunk 3, Chunk 9]", "chunks 2, 6, and 8"
CITATION = re.compile(
    r"\bchunks?\s*#?\s*(\d+(?:\s*(?:,\s*and|,|and|&)\s*(?:chunks?\s*#?\s*)?\d+)*)", re.IGNORECASE)


def cited_chunk_ids(answer):
    """Chunk numbers an answer cites, in order of first mention."""
    ids = []
    for match in CITATION.finditer(answer or ""):
        for number in re.findall(r"\d+", match.group(1)):
            if int(number) not in ids:
                ids.append(int(number))
    return ids


def citation_accuracy(cited_ids, evidence_chunk_ids):
    """Share of cited chunks that contain one of the question's evidence quotes; None if nothing is cited."""
    if not cited_ids:
        return None
    evidence = set(evidence_chunk_ids)
    return sum(1 for chunk_id in cited_ids if chunk_id in evidence) / len(cited_ids)


def faithfulness_score(claims):
    """Share of claims marked supported. None when the answer made no factual claims."""
    if not claims:
        return None
    return sum(1 for c in claims if c["supported"]) / len(claims)


def classify(answerable, gold_in_context, faithfulness, relevance=None,
             correctness=None, abstained=None):
    """Return (status, reason) for one question after the full pipeline ran.

    A failed answerable question is a retrieval miss when none of its gold
    chunks reached the model, and a generation problem when at least one did.
    An unanswerable question can only fail at generation.
    """
    problems = []
    if faithfulness is not None and faithfulness < FAITHFULNESS_MIN:
        problems.append(f"unsupported claims (faithfulness {faithfulness:.2f})")

    if not answerable:
        if not abstained:
            problems.append("answered a question the book can't answer")
        return (PASS, "") if not problems else (GENERATION_PROBLEM, "; ".join(problems))

    if relevance is not None and relevance < RELEVANCE_MIN:
        problems.append(f"low relevance ({relevance}/5)")
    if correctness == "incorrect":
        problems.append("contradicts or misses the reference answer")

    if not problems:
        return PASS, ""
    if not gold_in_context:
        return RETRIEVAL_MISS, "; ".join(["no gold chunk in the retrieved context"] + problems)
    return GENERATION_PROBLEM, "; ".join(problems)
