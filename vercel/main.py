"""Vercel-hosted read-only dashboard for summaries saved in TiDB."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import UUID

import certifi
import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

DATABASE_NAME = "steelers_analytics"
logger = logging.getLogger(__name__)
app = FastAPI(title="Steelers Analytics")


def _connection_options() -> dict[str, object]:
    connection_url = os.getenv("TIDB_DATABASE_URL")
    if connection_url:
        parsed = urlparse(connection_url)
        if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
            raise HTTPException(
                status_code=503,
                detail="TIDB_DATABASE_URL must use the mysql:// scheme.",
            )
        username = unquote(parsed.username or "")
        password = unquote(parsed.password or "")
        host = parsed.hostname or ""
        port = parsed.port or 4000
        database = unquote(parsed.path.lstrip("/")) or DATABASE_NAME
    else:
        username = os.getenv("TIDB_USER", "")
        password = os.getenv("TIDB_PASSWORD", "")
        host = os.getenv("TIDB_HOST", "")
        try:
            port = int(os.getenv("TIDB_PORT", "4000"))
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="TIDB_PORT must be a valid port number.",
            ) from exc
        database = DATABASE_NAME

    if not username or not password or not host:
        raise HTTPException(
            status_code=503,
            detail=(
                "TiDB is not configured. Set TIDB_DATABASE_URL or "
                "TIDB_USER, TIDB_PASSWORD, and TIDB_HOST in Vercel's "
                "environment variables."
            ),
        )
    if database != DATABASE_NAME:
        raise HTTPException(
            status_code=503,
            detail=f"TIDB_DATABASE_URL must select the {DATABASE_NAME} database.",
        )
    return {
        "host": host,
        "port": port,
        "user": username,
        "password": password,
        "database": DATABASE_NAME,
        "ssl_ca": certifi.where(),
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
        "connection_timeout": 10,
    }


def _query_rows(
    query: str, params: tuple[object, ...] = ()
) -> list[dict[str, object]]:
    try:
        connection = mysql.connector.connect(**_connection_options())
    except HTTPException:
        raise
    except mysql.connector.Error as exc:
        logger.exception("Could not connect to TiDB")
        raise HTTPException(
            status_code=503,
            detail="Could not connect to TiDB. Check Vercel variables and TiDB network access.",
        ) from exc

    try:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    except mysql.connector.Error as exc:
        logger.exception("TiDB dashboard query failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not read the dashboard tables from TiDB. Save all "
                "summaries from the local app first, then check database permissions."
            ),
        ) from exc
    finally:
        connection.close()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """Serve the Vercel dashboard page."""
    page_path = Path(__file__).with_name("index.html")
    return page_path.read_text(encoding="utf-8")


@app.get("/api/runs")
def list_runs() -> dict[str, list[dict[str, object]]]:
    """Return recent saved dashboard runs."""
    runs = _query_rows(
        """
        SELECT run_id, seasons, down_filter, MAX(saved_at) AS saved_at
        FROM touchdown_summaries
        GROUP BY run_id, seasons, down_filter
        ORDER BY saved_at DESC
        LIMIT 50
        """
    )
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    """Return all three saved summaries for one validated run ID."""
    try:
        canonical_run_id = str(UUID(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run ID.") from exc

    touchdown = _query_rows(
        """
        SELECT distance_band AS situation, play_category, plays, touchdowns,
               touchdown_rate
        FROM touchdown_summaries
        WHERE run_id = %s
        ORDER BY FIELD(distance_band, '1-5', '6-10', '11-20', '21-30',
                       '31-40', '41-50', '50+'),
                 play_category
        """,
        (canonical_run_id,),
    )
    if not touchdown:
        raise HTTPException(status_code=404, detail="Saved run not found.")

    turnover_outcomes = _query_rows(
        """
        SELECT turnover_margin, games, wins, losses, ties, win_rate
        FROM turnover_margin_outcomes
        WHERE run_id = %s
        ORDER BY FIELD(turnover_margin, 'Lost by 3+', 'Lost by 2',
                       'Lost by 1', 'Even', 'Won by 1', 'Won by 2',
                       'Won by 3+')
        """,
        (canonical_run_id,),
    )
    defensive_turnovers = _query_rows(
        """
        SELECT blitz_pressure, pass_plays, interceptions, strip_sacks_forced,
               strip_sacks_recovered, interception_rate, strip_sack_rate
        FROM defensive_turnovers_by_blitz
        WHERE run_id = %s
        ORDER BY FIELD(blitz_pressure, 'No charted blitzers',
                       '1 charted blitzer', '2+ charted blitzers',
                       'Blitz count unavailable')
        """,
        (canonical_run_id,),
    )
    return {
        "run_id": canonical_run_id,
        "touchdown_summary": touchdown,
        "turnover_outcomes": turnover_outcomes,
        "defensive_turnovers": defensive_turnovers,
    }
