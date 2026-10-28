import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from nfl_data import load_ftn_charting, load_play_by_play


class LocalNflDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temporary_directory.name)
        self.pbp = pl.DataFrame({"season": [2024], "play_id": [1]})
        self.ftn = pl.DataFrame(
            {
                "season": [2024],
                "nflverse_game_id": ["game1"],
                "nflverse_play_id": [1],
                "is_screen_pass": [True],
            }
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_play_by_play_is_saved_and_reused(self) -> None:
        with (
            patch("nfl_data.DATA_DIR", self.data_directory),
            patch("nfl_data.nfl.load_pbp", return_value=self.pbp) as load_pbp,
        ):
            first = load_play_by_play((2024,))
            second = load_play_by_play((2024,))

        self.assertTrue(first.equals(self.pbp))
        self.assertTrue(second.equals(self.pbp))
        load_pbp.assert_called_once_with(2024)
        self.assertTrue((self.data_directory / "play_by_play_2024.parquet").exists())

    def test_ftn_charting_is_saved_and_reused(self) -> None:
        with (
            patch("nfl_data.DATA_DIR", self.data_directory),
            patch(
                "nfl_data.nfl.load_ftn_charting", return_value=self.ftn
            ) as load_charting,
        ):
            first = load_ftn_charting((2024,))
            second = load_ftn_charting((2024,))

        self.assertTrue(first.equals(self.ftn))
        self.assertTrue(second.equals(self.ftn))
        load_charting.assert_called_once_with(2024)
        self.assertTrue((self.data_directory / "ftn_charting_2024.parquet").exists())

    def test_ftn_rejects_seasons_before_source_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "only available from 2022"):
            load_ftn_charting((2021,))


if __name__ == "__main__":
    unittest.main()
