import unittest

from eval import metrics


class RetrievalMetricTests(unittest.TestCase):
    def test_first_gold_rank_is_one_based(self):
        self.assertEqual(metrics.first_gold_rank([7, 3, 48, 75], [48, 75]), 3)

    def test_first_gold_rank_none_when_missing(self):
        self.assertIsNone(metrics.first_gold_rank([1, 2, 3], [48]))

    def test_hit_at_k(self):
        self.assertEqual(metrics.hit_at_k(3, 1), 0)
        self.assertEqual(metrics.hit_at_k(3, 3), 1)
        self.assertEqual(metrics.hit_at_k(None, 5), 0)

    def test_reciprocal_rank(self):
        self.assertEqual(metrics.reciprocal_rank(4), 0.25)
        self.assertEqual(metrics.reciprocal_rank(None), 0.0)

    def test_reciprocal_rank_respects_cutoff(self):
        self.assertEqual(metrics.reciprocal_rank(10, cutoff=15), 0.1)
        self.assertEqual(metrics.reciprocal_rank(208, cutoff=15), 0.0)

    def test_precision_at_k(self):
        retrieved = [7, 48, 3, 75, 9, 1, 2]
        self.assertEqual(metrics.precision_at_k(retrieved, [48, 75], 5), 0.4)
        self.assertEqual(metrics.precision_at_k(retrieved, [100], 5), 0.0)
        self.assertIsNone(metrics.precision_at_k(retrieved[:5], [48], 15))  # only a top 5 was saved

    def test_precision_ceiling(self):
        self.assertEqual(metrics.precision_ceiling([47], 5), 0.2)
        self.assertEqual(metrics.precision_ceiling([1, 2, 3], 15), 0.2)


class CitationTests(unittest.TestCase):
    def test_citation_formats_seen_in_answers(self):
        cases = {
            "He stole pears (Chunk 12).": [12],
            "Answer text\n[Citations: Chunk 3, Chunk 9]": [3, 9],
            "Monica wept (Chunks 47, 74).": [47, 74],
            "See (Chunks 4, 5, and 7) and (Citation: Chunk 4)": [4, 5, 7],
            "the themes in chunks 2, 6 and 8 about seeking God": [2, 6, 8],
            "(Cited from Chunk 160)": [160],
            "[Citations: None from the provided chunks]": [],
            "I don't know.": [],
        }
        for answer, expected in cases.items():
            self.assertEqual(metrics.cited_chunk_ids(answer), expected, answer)

    def test_citation_accuracy(self):
        self.assertEqual(metrics.citation_accuracy([47, 48, 74, 3], [47, 74]), 0.5)
        self.assertEqual(metrics.citation_accuracy([47], [47, 74]), 1.0)
        self.assertIsNone(metrics.citation_accuracy([], [47]))


class GenerationMetricTests(unittest.TestCase):
    def test_faithfulness_score(self):
        claims = [{"supported": True}, {"supported": True}, {"supported": False}, {"supported": True}]
        self.assertEqual(metrics.faithfulness_score(claims), 0.75)
        self.assertIsNone(metrics.faithfulness_score([]))

    def test_good_answer_passes(self):
        self.assertEqual(metrics.classify(True, True, 1.0, relevance=5, correctness="correct"), ("pass", ""))

    def test_bad_answer_without_gold_context_is_retrieval_miss(self):
        status, _ = metrics.classify(True, False, 1.0, relevance=1, correctness="incorrect")
        self.assertEqual(status, metrics.RETRIEVAL_MISS)

    def test_bad_answer_with_gold_context_is_generation_problem(self):
        status, reason = metrics.classify(True, True, 0.5, relevance=5, correctness="correct")
        self.assertEqual(status, metrics.GENERATION_PROBLEM)
        self.assertIn("faithfulness 0.50", reason)

    def test_unanswerable_must_abstain(self):
        self.assertEqual(metrics.classify(False, False, None, abstained=True), ("pass", ""))
        status, _ = metrics.classify(False, False, None, abstained=False)
        self.assertEqual(status, metrics.GENERATION_PROBLEM)


if __name__ == "__main__":
    unittest.main()
