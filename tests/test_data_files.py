"""Checks that prepared CSV files are written reproducibly."""

import tempfile
import unittest
from pathlib import Path

from banking77.data import read_records, write_records


class WriteRecordsTests(unittest.TestCase):
    def test_csv_uses_lf_line_endings_and_round_trips_embedded_newlines(self):
        rows = [
            {"id": "train-00000", "text": "\nWhere can I withdraw money from?", "category": "atm_support"},
            {"id": "train-00001", "text": "My card", "category": "card_arrival"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            write_records(path, rows)
            self.assertNotIn(b"\r", path.read_bytes())
            self.assertEqual(read_records(path), rows)


if __name__ == "__main__":
    unittest.main()
