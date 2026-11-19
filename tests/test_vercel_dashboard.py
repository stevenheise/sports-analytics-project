import unittest
from unittest.mock import patch

from fastapi import HTTPException

from vercel.main import app, dashboard, get_run, list_runs


class VercelDashboardTests(unittest.TestCase):
    def test_fastapi_app_exposes_dashboard_and_api_routes(self) -> None:
        paths = {route.path for route in app.routes}

        self.assertIn("/", paths)
        self.assertIn("/api/runs", paths)
        self.assertIn("/api/runs/{run_id}", paths)

    def test_dashboard_returns_the_tables_only_page(self) -> None:
        page = dashboard()

        self.assertIn("Touchdown rate by distance to goal", page)
        self.assertIn("Win rate by turnover margin", page)
        self.assertIn("Defensive takeaways by blitz count", page)
        self.assertIn("Player value leaderboard", page)
        self.assertIn('["tackles", "Tackles"]', page)
        self.assertIn('["sacks", "Sacks"]', page)
        self.assertIn('value="active">Currently active players', page)
        self.assertIn('data-page-link="players"', page)
        self.assertIn('id="touchdowns-page"', page)
        self.assertIn('id="turnovers-page"', page)
        self.assertIn('id="defense-page"', page)
        self.assertIn("Math.log(Number(record.cap_hit_m) * 1_000_000)", page)
        self.assertIn("older run; save a new run", page)
        self.assertNotIn("<canvas", page)

    @patch("vercel.main._query_rows")
    def test_list_runs_returns_saved_run_metadata(self, query_rows) -> None:
        runs = [{"run_id": "run-id", "seasons": "2024", "down_filter": None}]
        query_rows.return_value = runs

        result = list_runs()

        self.assertEqual(result, {"runs": runs})
        self.assertEqual(query_rows.call_count, 1)

    @patch("vercel.main._query_rows")
    def test_get_run_returns_all_summary_tables(self, query_rows) -> None:
        query_rows.side_effect = [
            [{"situation": "1-5", "play_category": "Inside run"}],
            [{"turnover_margin": "Even"}],
            [{"blitz_pressure": "No charted blitzers"}],
            [{"Field": "qb_td_int_ratio"}],
            [{
                "player_name": "Steelers player",
                "position_group": "QB",
                "qb_td_int_ratio": 2.0,
                "is_active": 1,
                "is_steelers": 1,
            }],
        ]
        run_id = "81bc7640-0000-4000-8000-000000000001"

        result = get_run(run_id.upper())

        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(len(result["touchdown_summary"]), 1)
        self.assertEqual(len(result["turnover_outcomes"]), 1)
        self.assertEqual(len(result["defensive_turnovers"]), 1)
        self.assertEqual(len(result["player_values"]), 1)
        self.assertTrue(result["qb_td_int_ratio_available"])
        self.assertEqual(query_rows.call_count, 5)
        self.assertEqual(query_rows.call_args_list[0].args[1], (run_id,))
        self.assertIn("SHOW COLUMNS", query_rows.call_args_list[3].args[0])
        self.assertIn("qb_td_int_ratio", query_rows.call_args_list[4].args[0])
        self.assertIn("tackles, sacks", query_rows.call_args_list[4].args[0])

    @patch("vercel.main._query_rows")
    def test_get_run_supports_saved_runs_before_qb_ratio_migration(
        self, query_rows
    ) -> None:
        query_rows.side_effect = [
            [{"situation": "1-5"}],
            [],
            [],
            [{"Field": "offensive_touchdowns"}],
            [],
        ]

        result = get_run("81bc7640-0000-4000-8000-000000000001")

        self.assertFalse(result["qb_td_int_ratio_available"])
        self.assertIn(
            "0 AS qb_td_int_ratio",
            query_rows.call_args_list[4].args[0],
        )
        self.assertIn("is_active, is_steelers", query_rows.call_args_list[4].args[0])

    @patch("vercel.main._query_rows")
    def test_get_run_flags_older_rows_after_qb_ratio_migration(
        self, query_rows
    ) -> None:
        query_rows.side_effect = [
            [{"situation": "1-5"}],
            [],
            [],
            [{"Field": "qb_td_int_ratio"}],
            [{"position_group": "QB", "qb_td_int_ratio": 0}],
        ]

        result = get_run("81bc7640-0000-4000-8000-000000000001")

        self.assertFalse(result["qb_td_int_ratio_available"])

    @patch("vercel.main._query_rows")
    def test_get_run_rejects_invalid_id_without_querying_database(self, query_rows) -> None:
        with self.assertRaises(HTTPException) as raised:
            get_run("not-a-uuid")

        self.assertEqual(raised.exception.status_code, 400)
        query_rows.assert_not_called()

    @patch("vercel.main._query_rows")
    def test_get_run_returns_404_for_missing_run(self, query_rows) -> None:
        query_rows.return_value = []

        with self.assertRaises(HTTPException) as raised:
            get_run("81bc7640-0000-4000-8000-000000000001")

        self.assertEqual(raised.exception.status_code, 404)
        query_rows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
