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
        ]
        run_id = "81bc7640-0000-4000-8000-000000000001"

        result = get_run(run_id.upper())

        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(len(result["touchdown_summary"]), 1)
        self.assertEqual(len(result["turnover_outcomes"]), 1)
        self.assertEqual(len(result["defensive_turnovers"]), 1)
        self.assertEqual(query_rows.call_count, 3)
        self.assertEqual(query_rows.call_args_list[0].args[1], (run_id,))

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
