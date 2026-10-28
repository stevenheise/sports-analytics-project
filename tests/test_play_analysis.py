import unittest
from unittest.mock import MagicMock, patch

import mysql.connector
import polars as pl

import database
from play_analysis import (
    PLAY_TYPES,
    SITUATIONS,
    summarize_defensive_turnovers_by_blitz,
    summarize_plays,
    summarize_turnover_margin_outcomes,
)


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

    @patch("database.mysql.connector.connect")
    def test_save_analysis_summaries_persists_all_tables_in_one_transaction(
        self, connect: MagicMock
    ) -> None:
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        connect.return_value = connection

        touchdown = summarize_plays(self.pbp, [2024], ftn=self.ftn)
        outcomes = pl.DataFrame(
            {
                "Turnover margin": ["Even"],
                "Games": [2],
                "Wins": [1],
                "Losses": [1],
                "Ties": [0],
                "Win rate (%)": [50.0],
            }
        )
        defensive = pl.DataFrame(
            {
                "Blitz pressure": ["No charted blitzers"],
                "Pass plays": [10],
                "Interceptions": [1],
                "Strip sacks (forced)": [2],
                "Strip sacks recovered": [1],
                "Interception rate (%)": [10.0],
                "Strip-sack rate (%)": [20.0],
            }
        )

        run_id = database.save_analysis_summaries(
            touchdown, outcomes, defensive, [2023, 2024], down=1
        )

        self.assertTrue(run_id)
        self.assertEqual(cursor.execute.call_count, 3)
        self.assertEqual(cursor.executemany.call_count, 3)
        for saved_rows in (
            cursor.executemany.call_args_list[0].args[1],
            cursor.executemany.call_args_list[1].args[1],
            cursor.executemany.call_args_list[2].args[1],
        ):
            self.assertTrue(all(row[0] == run_id for row in saved_rows))
            self.assertTrue(all(row[1] == "2023,2024" for row in saved_rows))
        self.assertEqual(
            len(cursor.executemany.call_args_list[0].args[1]), touchdown.height
        )
        self.assertEqual(
            cursor.executemany.call_args_list[1].args[1][0][2], "Even"
        )
        self.assertEqual(
            cursor.executemany.call_args_list[2].args[1][0][2],
            "No charted blitzers",
        )
        connection.start_transaction.assert_called_once()
        connection.commit.assert_called_once()
        connection.close.assert_called_once()

    @patch("database.mysql.connector.connect")
    def test_save_analysis_summaries_rolls_back_on_insert_error(
        self, connect: MagicMock
    ) -> None:
        connection = MagicMock()
        cursor = MagicMock()
        cursor.executemany.side_effect = mysql.connector.Error("write failed")
        connection.cursor.return_value.__enter__.return_value = cursor
        connect.return_value = connection

        touchdown = summarize_plays(self.pbp, [2024], ftn=self.ftn)
        outcomes = pl.DataFrame(
            {
                "Turnover margin": ["Even"],
                "Games": [1],
                "Wins": [1],
                "Losses": [0],
                "Ties": [0],
                "Win rate (%)": [100.0],
            }
        )
        defensive = pl.DataFrame(
            {
                "Blitz pressure": ["No charted blitzers"],
                "Pass plays": [1],
                "Interceptions": [0],
                "Strip sacks (forced)": [0],
                "Strip sacks recovered": [0],
                "Interception rate (%)": [0.0],
                "Strip-sack rate (%)": [0.0],
            }
        )

        with self.assertRaises(mysql.connector.Error):
            database.save_analysis_summaries(
                touchdown, outcomes, defensive, [2024], down=None
            )

        connection.rollback.assert_called_once()
        connection.close.assert_called_once()


class TurnoverMarginOutcomeTests(unittest.TestCase):
    def test_summarizes_win_rates_by_turnover_margin(self) -> None:
        rows = []
        games = [
            ("win_by_one", 2024, "PIT", "BAL", 24, 20, 1, 2),
            ("loss_by_one", 2024, "PIT", "CLE", 10, 17, 2, 1),
            ("tie_even", 2024, "PIT", "CIN", 14, 14, 0, 0),
            ("win_by_three_plus", 2024, "PIT", "HOU", 31, 10, 0, 3),
            ("other_season", 2023, "PIT", "NYJ", 20, 10, 0, 0),
        ]
        for game_id, season, home, away, home_score, away_score, pit_to, opp_to in games:
            for team, turnovers in [("PIT", pit_to), (away, opp_to)]:
                for index in range(max(turnovers, 1)):
                    rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "home_team": home,
                            "away_team": away,
                            "home_score": home_score,
                            "away_score": away_score,
                            "posteam": team,
                            "play_type": "pass",
                            "interception": int(index < turnovers),
                            "fumble_lost": 0,
                        }
                    )
        result = summarize_turnover_margin_outcomes(
            pl.DataFrame(rows), [2024]
        )
        by_margin = {
            row["Turnover margin"]: row for row in result.to_dicts()
        }

        self.assertEqual(result.height, 7)
        self.assertEqual(by_margin["Won by 1"]["Games"], 1)
        self.assertEqual(by_margin["Won by 1"]["Wins"], 1)
        self.assertEqual(by_margin["Won by 1"]["Win rate (%)"], 100.0)
        self.assertEqual(by_margin["Lost by 1"]["Losses"], 1)
        self.assertEqual(by_margin["Lost by 1"]["Win rate (%)"], 0.0)
        self.assertEqual(by_margin["Even"]["Ties"], 1)
        self.assertEqual(by_margin["Even"]["Win rate (%)"], 0.0)
        self.assertEqual(by_margin["Won by 3+"]["Wins"], 1)
        self.assertEqual(by_margin["Lost by 2"]["Games"], 0)
        self.assertIsNone(by_margin["Lost by 2"]["Win rate (%)"])


class DefensiveTurnoverByBlitzTests(unittest.TestCase):
    def test_summarizes_takeaways_by_charted_blitz_count(self) -> None:
        pbp = pl.DataFrame(
            {
                "game_id": ["g1", "g1", "g1", "g1", "g2"],
                "play_id": [1, 2, 3, 4, 5],
                "season": [2024, 2024, 2024, 2024, 2021],
                "posteam": ["BAL", "BAL", "BAL", "BAL", "CLE"],
                "defteam": ["PIT"] * 5,
                "play_type": ["pass"] * 5,
                "interception": [1, 0, 0, 0, 1],
                "sack": [0, 1, 1, 0, 0],
                "fumble_forced": [0, 1, 1, 0, 0],
                "fumble_lost": [0, 1, 0, 0, 0],
            }
        )
        ftn = pl.DataFrame(
            {
                "nflverse_game_id": ["g1"] * 4,
                "nflverse_play_id": [1, 2, 3, 4],
                "n_blitzers": [0, 1, 2, 0],
            }
        )

        result = summarize_defensive_turnovers_by_blitz(pbp, [2021, 2024], ftn)
        by_blitz = {
            row["Blitz pressure"]: row for row in result.to_dicts()
        }

        self.assertEqual(result.height, 4)
        self.assertEqual(by_blitz["No charted blitzers"]["Pass plays"], 2)
        self.assertEqual(by_blitz["No charted blitzers"]["Interceptions"], 1)
        self.assertEqual(
            by_blitz["No charted blitzers"]["Interception rate (%)"], 50.0
        )
        self.assertEqual(
            by_blitz["1 charted blitzer"]["Strip sacks (forced)"], 1
        )
        self.assertEqual(
            by_blitz["1 charted blitzer"]["Strip sacks recovered"], 1
        )
        self.assertEqual(
            by_blitz["2+ charted blitzers"]["Strip sacks recovered"], 0
        )
        self.assertEqual(
            by_blitz["Blitz count unavailable"]["Interceptions"], 1
        )
        self.assertEqual(
            by_blitz["Blitz count unavailable"]["Pass plays"], 1
        )

    def test_missing_ftn_data_keeps_pass_plays_unavailable(self) -> None:
        pbp = pl.DataFrame(
            {
                "game_id": ["g1"],
                "play_id": [1],
                "season": [2021],
                "posteam": ["BAL"],
                "defteam": ["PIT"],
                "play_type": ["pass"],
                "interception": [0],
                "sack": [0],
                "fumble_forced": [0],
                "fumble_lost": [0],
            }
        )
        result = summarize_defensive_turnovers_by_blitz(pbp, [2021])
        unavailable = result.filter(
            pl.col("Blitz pressure") == "Blitz count unavailable"
        ).row(0, named=True)

        self.assertEqual(unavailable["Pass plays"], 1)
        self.assertEqual(unavailable["Interception rate (%)"], 0.0)


if __name__ == "__main__":
    unittest.main()
