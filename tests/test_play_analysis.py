import unittest
from unittest.mock import MagicMock, patch

import mysql.connector
import polars as pl

import database
from play_analysis import PLAY_TYPES, SITUATIONS, summarize_plays


class SummarizePlaysTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pbp = pl.DataFrame(
            {
                "posteam": [
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "PIT",
                    "BAL",
                ],
                "season": [2024] * 10,
                "down": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
                "yardline_100": [5, 50, 10, 51, 5, 25, 15, 35, 15, 5],
                "play_type": [
                    "pass",
                    "run",
                    "pass",
                    "run",
                    "run",
                    "pass",
                    "pass",
                    "pass",
                    "run",
                    "pass",
                ],
                "pass_touchdown": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "rush_touchdown": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "run_location": [
                    None,
                    "right",
                    None,
                    "left",
                    "middle",
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "pass_location": [
                    "middle",
                    None,
                    "middle",
                    None,
                    None,
                    "left",
                    "right",
                    None,
                    None,
                    "left",
                ],
                "game_id": ["game1"] * 10,
                "play_id": list(range(1, 11)),
            }
        )
        self.ftn = pl.DataFrame(
            {
                "nflverse_game_id": ["game1"] * 6,
                "nflverse_play_id": [1, 2, 3, 4, 5, 6],
                "is_screen_pass": [True, False, False, False, False, False],
            }
        )

    def test_summarizes_categories_and_touchdown_rates(self) -> None:
        result = summarize_plays(self.pbp, [2024], ftn=self.ftn)
        five_yards = result.filter(pl.col("Situation") == "1-5")
        screen = five_yards.filter(pl.col("Play type") == "Screen pass").row(
            0, named=True
        )
        inside = five_yards.filter(pl.col("Play type") == "Inside run").row(
            0, named=True
        )

        self.assertEqual(result.height, len(SITUATIONS) * len(PLAY_TYPES))
        self.assertEqual(screen["Plays"], 1)
        self.assertEqual(screen["Touchdowns"], 1)
        self.assertEqual(screen["Touchdown rate (%)"], 100.0)
        self.assertEqual(inside["Plays"], 1)
        self.assertEqual(inside["Touchdowns"], 0)
        self.assertEqual(inside["Touchdown rate (%)"], 0.0)

    def test_location_categories_distance_bands_and_down_filter(self) -> None:
        result = summarize_plays(self.pbp, [2024], down=1, ftn=self.ftn)
        by_category = {
            (row["Situation"], row["Play type"]): row
            for row in result.to_dicts()
        }
        self.assertEqual(by_category[("6-10", "Pass - middle")]["Plays"], 0)
        self.assertEqual(by_category[("41-50", "Outside run")]["Plays"], 1)
        self.assertEqual(by_category[("50+", "Outside run")]["Touchdown rate (%)"], 100.0)
        self.assertEqual(
            by_category[("21-30", "Pass - sideline")]["Touchdown rate (%)"], 100.0
        )
        self.assertEqual(
            by_category[("11-20", "Run (location unavailable)")]["Plays"], 1
        )
        self.assertEqual(
            by_category[("11-20", "Pass (classification unavailable)")]["Plays"], 1
        )

    def test_missing_ftn_keeps_passes_unclassified(self) -> None:
        result = summarize_plays(self.pbp, [2024])
        unclassified_passes = result.filter(
            pl.col("Play type") == "Pass (classification unavailable)"
        )
        self.assertEqual(unclassified_passes.get_column("Plays").sum(), 5)
        self.assertEqual(result.get_column("Plays").sum(), 9)

    def test_non_steelers_and_unselected_seasons_are_excluded(self) -> None:
        result = summarize_plays(self.pbp, [2023], ftn=self.ftn)
        self.assertEqual(result.get_column("Plays").sum(), 0)
        self.assertEqual(result.filter(pl.col("Plays") == 0).height, 49)

    def test_missing_required_columns_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            summarize_plays(pl.DataFrame({"posteam": ["PIT"]}), [2024])

    @patch("database.mysql.connector.connect")
    def test_save_writes_every_summary_row_in_one_transaction(
        self, connect: MagicMock
    ) -> None:
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        connect.return_value = connection

        result = summarize_plays(self.pbp, [2024], ftn=self.ftn)
        run_id = database.save_touchdown_summary(result, [2024], down=1)

        self.assertTrue(run_id)
        connect.assert_called_once()
        options = connect.call_args.kwargs
        self.assertEqual(options["database"], "steelers_analytics")
        self.assertTrue(options["ssl_verify_cert"])
        self.assertTrue(options["ssl_verify_identity"])
        self.assertEqual(cursor.executemany.call_count, 1)
        saved_rows = cursor.executemany.call_args.args[1]
        self.assertEqual(len(saved_rows), result.height)
        self.assertTrue(all(row[0] == run_id for row in saved_rows))
        self.assertTrue(all(row[2] == 1 for row in saved_rows))
        connection.commit.assert_called_once()
        connection.close.assert_called_once()

    @patch("database.mysql.connector.connect")
    def test_save_surfaces_database_errors_and_rolls_back(
        self, connect: MagicMock
    ) -> None:
        connection = MagicMock()
        cursor = MagicMock()
        cursor.executemany.side_effect = mysql.connector.Error("write failed")
        connection.cursor.return_value.__enter__.return_value = cursor
        connect.return_value = connection

        result = summarize_plays(self.pbp, [2024], ftn=self.ftn)
        with self.assertRaises(mysql.connector.Error):
            database.save_touchdown_summary(result, [2024], down=None)

        connection.rollback.assert_called_once()
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
