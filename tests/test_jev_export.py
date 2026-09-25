import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from sokoban_state_modeling.sft.jev_export import build_record, export_jev


def source(grid=None, split="train"):
    return {"current_state": {"grid": grid or ["#######", "#P_XO_#", "#_*___#", "#######"]},
            "split": split, "task_id": "example", "query_actions": ["Left"],
            "target": "SECRET_OLD_THINKING"}


class JevExportTests(unittest.TestCase):
    def test_seven_classes_and_no_label_leakage(self):
        grids = [["#######", "#P_XO_#", "#_*___#", "#######"],
                 ["#####", "#+X_#", "#####"]]
        found = set()
        for grid in grids:
            record, _ = build_record(source(grid))
            self.assertNotIn("SECRET", json.dumps(record))
            self.assertNotIn("probabilities", json.dumps(record["input"]))
            for r, row in enumerate(grid):
                for c, symbol in enumerate(row):
                    probs = record["target"]["answers"][f"cell_{r}_{c}"]["probabilities"]
                    self.assertEqual(sum(probs.values()), 1)
                    expected = {"#": "wall", "_": "floor", "O": "goal", "X": "box",
                                "*": "box_on_goal", "P": "player", "+": "player_on_goal"}[symbol]
                    self.assertEqual(probs[expected], 1)
                    found.add(expected)
        self.assertEqual(len(found), 7)

    def test_export_images_dedup_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "sft.jsonl"
            src.write_text((json.dumps(source()) + "\n") * 2)
            out = root / "export"
            report = export_jev([src], out)
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["duplicates_removed"], 1)
            record = json.loads((out / "records.jsonl").read_text())
            with Image.open(out / record["input"]["image_ref"]) as image:
                self.assertEqual(image.size, (7 * 16, 4 * 16))
            with self.assertRaises(FileExistsError):
                export_jev([src], out)

    def test_cross_split_mirror_leakage_fails_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = source()
            second = source([row[::-1] for row in first["current_state"]["grid"]], "test")
            src = root / "sft.jsonl"
            src.write_text("\n".join(map(json.dumps, [first, second])))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                export_jev([src], root / "out")
            self.assertFalse((root / "out").exists())

    def test_large_board_groups_and_corruption(self):
        grid = ["#########", "#PXO____#"] + ["#_______#"] * 6 + ["#########"]
        record, _ = build_record(source(grid))
        self.assertEqual(list(map(len, record["input"]["request_groups"])), [64, 17])
        bad = source()
        bad["current_state_hash"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_record(bad)


if __name__ == "__main__":
    unittest.main()
