import csv
import tempfile
import unittest
from pathlib import Path

from eval import run_eval


class IncrementalSaveTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "results.csv"

    def test_rows_are_appended_under_one_header(self):
        run_eval.append_row({"id": "q01", "answerable": True, "question": "First?"}, self.path)
        run_eval.append_row({"id": "q02", "answerable": False, "question": "Second?"}, self.path)

        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, run_eval.CSV_COLUMNS)
        self.assertEqual([r["id"] for r in rows], ["q01", "q02"])
        self.assertEqual(rows[1]["answerable"], "False")

    def test_saved_ids_lists_finished_questions(self):
        self.assertEqual(run_eval.saved_ids(self.path), set())  # no file yet
        run_eval.append_row({"id": "q07"}, self.path)
        run_eval.append_row({"id": "q09"}, self.path)
        self.assertEqual(run_eval.saved_ids(self.path), {"q07", "q09"})

    def test_saved_ids_refuses_a_csv_with_other_columns(self):
        self.path.write_text("id,answer\nq01,yes\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            run_eval.saved_ids(self.path)


if __name__ == "__main__":
    unittest.main()
