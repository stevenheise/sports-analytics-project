import unittest

import polars as pl

from player_analysis import (
    current_steelers_roster_ids,
    summarize_player_value,
)


class SummarizePlayerValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = pl.DataFrame(
            {
                "player_id": ["qb1", "wr1", "db1", "rb1"],
                "player_display_name": ["Quarterback", "Receiver", "Defender", "Unpriced"],
                "position_group": ["QB", "WR", "DB", "RB"],
                "season": [2024, 2024, 2024, 2024],
                "recent_team": ["CIN", "2TM", "PIT", "PIT"],
                "games": [12, 10, 15, 1],
                "passing_yards": [2000, 0, 0, 0],
                "rushing_yards": [100, 0, 0, 0],
                "receiving_yards": [0, 500, 0, 25],
                "passing_tds": [18, 0, 0, 0],
                "rushing_tds": [2, 0, 0, 0],
                "receiving_tds": [0, 5, 0, 0],
                "def_interceptions": [0, 0, 2, 0],
                "def_fumbles_forced": [0, 0, 1, 0],
                "fumble_recovery_opp": [0, 0, 1, 0],
                "def_tackles_solo": [0, 0, 10, 0],
                "def_tackle_assists": [0, 0, 5, 0],
                "def_sacks": [0, 0, 3, 0],
            }
        )
        history_type = pl.List(
            pl.Struct(
                {
                    "year": pl.String,
                    "team": pl.String,
                    "cap_number": pl.Float64,
                }
            )
        )
        self.contracts = pl.DataFrame(
            {
                "gsis_id": ["qb1", "qb1", "wr1", "db1"],
                "season_history": pl.Series(
                    "season_history",
                    [
                        [{"year": "2024", "team": "Bengals", "cap_number": 10.0}],
                        [{"year": "2024", "team": "Bengals", "cap_number": 10.0}],
                        [{"year": "2024", "team": "Steelers", "cap_number": 5.0}],
                        [{"year": "2024", "team": "Steelers", "cap_number": 4.0}],
                    ],
                    dtype=history_type,
                ),
            }
        )

    def test_summarizes_role_stats_and_deduplicates_cap_history(self) -> None:
        result = summarize_player_value(
            self.stats,
            self.contracts,
            [2024],
            current_steelers_ids={"db1"},
        )
        by_player = {row["player_id"]: row for row in result.to_dicts()}

        self.assertEqual(set(by_player), {"qb1", "wr1", "db1"})
        self.assertEqual(by_player["qb1"]["cap_hit_m"], 10.0)
        self.assertEqual(by_player["qb1"]["offensive_yards"], 2100.0)
        self.assertEqual(by_player["qb1"]["offensive_touchdowns"], 20.0)
        self.assertEqual(by_player["wr1"]["offensive_yards"], 500.0)
        self.assertFalse(by_player["wr1"]["is_steelers"])
        self.assertEqual(by_player["db1"]["defensive_takeaways"], 3.0)
        self.assertEqual(by_player["db1"]["forced_fumbles"], 1.0)
        self.assertEqual(by_player["db1"]["tackles"], 15.0)
        self.assertEqual(by_player["db1"]["sacks"], 3.0)
        self.assertTrue(by_player["db1"]["is_steelers"])

    def test_current_steelers_roster_uses_latest_week_and_roster_statuses(self) -> None:
        rosters = pl.DataFrame(
            {
                "week": [10, 11, 11, 11, 11],
                "team": ["PIT", "PIT", "PIT", "PIT", "CIN"],
                "status": ["ACT", "ACT", "RES", "CUT", "ACT"],
                "gsis_id": ["past", "current", "reserve", "cut", "other"],
            }
        )

        self.assertEqual(
            current_steelers_roster_ids(rosters),
            {"current", "reserve"},
        )

    def test_empty_seasons_returns_typed_empty_frame(self) -> None:
        result = summarize_player_value(self.stats, self.contracts, [])

        self.assertEqual(result.height, 0)
        self.assertIn("cap_hit_m", result.columns)


if __name__ == "__main__":
    unittest.main()