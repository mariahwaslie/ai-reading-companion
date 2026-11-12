import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from core import vectorstore


class FakeEncoder:
    """Stands in for SentenceTransformer so the test needs no model: the same text always gets the same vector."""

    def __init__(self, model_name):
        pass

    def encode(self, texts, **kwargs):
        digests = [hashlib.sha256(text.encode("utf-8")).digest() for text in texts]
        return np.array([[byte / 255 for byte in d[:8]] for d in digests], dtype=np.float32)


class SessionIsolationTests(unittest.TestCase):
    """app.py keeps one VectorStore per Streamlit session, so two VectorStores stand in for two users.

    In-memory Chroma clients in one process share their collections, which is how one user's
    upload used to replace everyone's book.
    """

    def setUp(self):
        patcher = patch.object(vectorstore, "SentenceTransformer", FakeEncoder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_session_gets_its_own_collection(self):
        first, second = vectorstore.VectorStore(), vectorstore.VectorStore()
        self.assertNotEqual(first.collection_name, second.collection_name)

    def test_two_sessions_do_not_see_each_others_chunks(self):
        alice = vectorstore.VectorStore()
        alice.build_index(["Alice's book, chapter one.", "Alice's book, chapter two."])
        bob = vectorstore.VectorStore()
        bob.build_index(["Bob's book, chapter one.", "Bob's book, chapter two.", "Bob's book, chapter three."])

        alice_results = alice.search("chapter one", k=2)["documents"]
        bob_results = bob.search("chapter one", k=3)["documents"]

        self.assertEqual(sorted(alice_results), ["Alice's book, chapter one.", "Alice's book, chapter two."])
        self.assertEqual(len(bob_results), 3)
        self.assertTrue(all(doc.startswith("Bob's book") for doc in bob_results))

    def test_new_upload_in_one_session_leaves_the_other_alone(self):
        alice = vectorstore.VectorStore()
        alice.build_index(["Alice's book, chapter one.", "Alice's book, chapter two."])
        bob = vectorstore.VectorStore()
        bob.build_index(["Bob's first book."])

        bob.build_index(["Bob's second book."])  # a re-upload clears only Bob's own collection

        self.assertEqual(alice.collection.count(), 2)
        self.assertEqual(bob.search("book", k=1)["documents"], ["Bob's second book."])


if __name__ == "__main__":
    unittest.main()
