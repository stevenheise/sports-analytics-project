"""TiDB persistence for generated Steelers analysis summaries."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse
from uuid import uuid4

import certifi
import mysql.connector
import polars as pl
from dotenv import load_dotenv
from mysql.connector import errorcode

DATABASE_NAME = "steelers_analytics"
TABLE_NAME = "touchdown_summaries"
TURNOVER_OUTCOMES_TABLE_NAME = "turnover_margin_outcomes"
DEFENSIVE_TURNOVERS_TABLE_NAME = "defensive_turnovers_by_blitz"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    run_id CHAR(36) NOT NULL,
    seasons VARCHAR(255) NOT NULL,
    down_filter TINYINT NULL,
    distance_band VARCHAR(20) NOT NULL,
    play_category VARCHAR(64) NOT NULL,
    plays INT UNSIGNED NOT NULL,
    touchdowns INT UNSIGNED NULL,
    touchdown_rate DECIMAL(5, 1) NULL,
    saved_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, distance_band, play_category),
    INDEX idx_touchdown_summaries_saved_at (saved_at)
)
"""
CREATE_TURNOVER_OUTCOMES_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TURNOVER_OUTCOMES_TABLE_NAME} (
    run_id CHAR(36) NOT NULL,
    seasons VARCHAR(255) NOT NULL,
    turnover_margin VARCHAR(32) NOT NULL,
    games INT UNSIGNED NOT NULL,
    wins INT UNSIGNED NOT NULL,
    losses INT UNSIGNED NOT NULL,
    ties INT UNSIGNED NOT NULL,
    win_rate DECIMAL(5, 1) NULL,
    saved_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, turnover_margin),
    INDEX idx_turnover_margin_outcomes_saved_at (saved_at)
)
"""
CREATE_DEFENSIVE_TURNOVERS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DEFENSIVE_TURNOVERS_TABLE_NAME} (
    run_id CHAR(36) NOT NULL,
    seasons VARCHAR(255) NOT NULL,
    blitz_pressure VARCHAR(40) NOT NULL,
    pass_plays INT UNSIGNED NOT NULL,
    interceptions INT UNSIGNED NOT NULL,
    strip_sacks_forced INT UNSIGNED NOT NULL,
    strip_sacks_recovered INT UNSIGNED NOT NULL,
    interception_rate DECIMAL(5, 1) NULL,
    strip_sack_rate DECIMAL(5, 1) NULL,
    saved_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id, blitz_pressure),
    INDEX idx_defensive_turnovers_by_blitz_saved_at (saved_at)
)
"""


class DatabaseConfigurationError(ValueError):
    """Raised when TiDB connection details are missing or invalid."""


def _connection_options(database: str) -> dict[str, object]:
    load_dotenv()
    connection_url = os.getenv("TIDB_DATABASE_URL")
    if connection_url:
        parsed = urlparse(connection_url)
        if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
            raise DatabaseConfigurationError(
                "TIDB_DATABASE_URL must use the mysql:// scheme."
            )
        username = unquote(parsed.username or "")
        url_password = unquote(parsed.password or "")
        password = (
            url_password
            if url_password and url_password != "[Password]"
            else os.getenv("TIDB_PASSWORD", os.getenv("data_base_password", ""))
        )
        host = parsed.hostname or ""
        port = parsed.port or 4000
    else:
        username = os.getenv("TIDB_USER", "3TPNC1.root")
        password = os.getenv("TIDB_PASSWORD") or os.getenv("data_base_password", "")
        host = os.getenv(
            "TIDB_HOST", "gateway01.us-east-1.prod.aws.tidbcloud.com"
        )
        port = int(os.getenv("TIDB_PORT", "4000"))

    if not username or not host or not password or password == "[Password]":
        raise DatabaseConfigurationError(
            "TiDB credentials are not configured. Set TIDB_DATABASE_URL or "
            "TIDB_USER, TIDB_PASSWORD, and TIDB_HOST in the local .env file."
        )

    return {
        "host": host,
        "port": port,
        "user": username,
        "password": password,
        "database": database,
        "ssl_ca": certifi.where(),
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
        "connection_timeout": 15,
    }


def _connect_to_database() -> mysql.connector.MySQLConnection:
    try:
        return mysql.connector.connect(**_connection_options(DATABASE_NAME))
    except mysql.connector.Error as exc:
        if exc.errno != errorcode.ER_BAD_DB_ERROR:
            raise

    bootstrap = mysql.connector.connect(**_connection_options("sys"))
    try:
        with bootstrap.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
    finally:
        bootstrap.close()
    return mysql.connector.connect(**_connection_options(DATABASE_NAME))


def save_touchdown_summary(
    summary: pl.DataFrame,
    seasons: list[int],
    down: int | None,
) -> str:
    """Persist all generated summary rows and return the saved run ID."""
    connection = _connect_to_database()
    run_id = str(uuid4())
    saved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    season_selection = ",".join(map(str, sorted(seasons)))
    rows = [
        (
            run_id,
            season_selection,
            down,
            row["Situation"],
            row["Play type"],
            row["Plays"],
            row["Touchdowns"],
            row["Touchdown rate (%)"],
            saved_at,
        )
        for row in summary.to_dicts()
    ]
    insert_sql = f"""
    INSERT INTO {TABLE_NAME} (
        run_id, seasons, down_filter, distance_band, play_category,
        plays, touchdowns, touchdown_rate, saved_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
            connection.start_transaction()
            cursor.executemany(insert_sql, rows)
        connection.commit()
    except mysql.connector.Error:
        connection.rollback()
        raise
    finally:
        connection.close()
    return run_id


def save_analysis_summaries(
    touchdown_summary: pl.DataFrame,
    turnover_outcomes: pl.DataFrame,
    defensive_turnovers: pl.DataFrame,
    seasons: list[int],
    down: int | None,
) -> str:
    """Persist all dashboard summaries together and return their shared run ID."""
    connection = _connect_to_database()
    run_id = str(uuid4())
    saved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    season_selection = ",".join(map(str, sorted(seasons)))
    touchdown_rows = [
        (
            run_id,
            season_selection,
            down,
            row["Situation"],
            row["Play type"],
            row["Plays"],
            row["Touchdowns"],
            row["Touchdown rate (%)"],
            saved_at,
        )
        for row in touchdown_summary.to_dicts()
    ]
    turnover_rows = [
        (
            run_id,
            season_selection,
            row["Turnover margin"],
            row["Games"],
            row["Wins"],
            row["Losses"],
            row["Ties"],
            row["Win rate (%)"],
            saved_at,
        )
        for row in turnover_outcomes.to_dicts()
    ]
    defensive_rows = [
        (
            run_id,
            season_selection,
            row["Blitz pressure"],
            row["Pass plays"],
            row["Interceptions"],
            row["Strip sacks (forced)"],
            row["Strip sacks recovered"],
            row["Interception rate (%)"],
            row["Strip-sack rate (%)"],
            saved_at,
        )
        for row in defensive_turnovers.to_dicts()
    ]
    inserts = [
        (
            CREATE_TABLE_SQL,
            f"""
            INSERT INTO {TABLE_NAME} (
                run_id, seasons, down_filter, distance_band, play_category,
                plays, touchdowns, touchdown_rate, saved_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            touchdown_rows,
        ),
        (
            CREATE_TURNOVER_OUTCOMES_TABLE_SQL,
            f"""
            INSERT INTO {TURNOVER_OUTCOMES_TABLE_NAME} (
                run_id, seasons, turnover_margin, games, wins, losses, ties,
                win_rate, saved_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            turnover_rows,
        ),
        (
            CREATE_DEFENSIVE_TURNOVERS_TABLE_SQL,
            f"""
            INSERT INTO {DEFENSIVE_TURNOVERS_TABLE_NAME} (
                run_id, seasons, blitz_pressure, pass_plays, interceptions,
                strip_sacks_forced, strip_sacks_recovered, interception_rate,
                strip_sack_rate, saved_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            defensive_rows,
        ),
    ]
    try:
        with connection.cursor() as cursor:
            for create_sql, _, _ in inserts:
                cursor.execute(create_sql)
            connection.start_transaction()
            for _, insert_sql, rows in inserts:
                cursor.executemany(insert_sql, rows)
        connection.commit()
    except mysql.connector.Error:
        connection.rollback()
        raise
    finally:
        connection.close()
    return run_id
