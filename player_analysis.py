"""Leaguewide player production and salary-cap efficiency summaries."""

from __future__ import annotations

import polars as pl

OFFENSIVE_POSITION_GROUPS = ("QB", "RB", "TE", "WR")
DEFENSIVE_POSITION_GROUPS = ("DB", "DL", "LB")

PLAYER_VALUE_SCHEMA = {
    "player_id": pl.String,
    "player_name": pl.String,
    "team": pl.String,
    "position_group": pl.String,
    "games": pl.Int64,
    "seasons": pl.Int64,
    "cap_hit_m": pl.Float64,
    "offensive_yards": pl.Float64,
    "offensive_touchdowns": pl.Float64,
    "tackles": pl.Float64,
    "sacks": pl.Float64,
    "defensive_interceptions": pl.Float64,
    "forced_fumbles": pl.Float64,
    "fumble_recoveries": pl.Float64,
    "defensive_takeaways": pl.Float64,
    "is_steelers": pl.Boolean,
}


def current_steelers_roster_ids(rosters: pl.DataFrame) -> set[str]:
    """Return player IDs on Pittsburgh's latest listed active/reserve roster."""
    required_columns = {"week", "team", "status", "gsis_id"}
    missing_columns = sorted(required_columns - set(rosters.columns))
    if missing_columns:
        raise ValueError(f"rosters are missing required columns: {missing_columns}")
    if rosters.is_empty():
        raise ValueError("the current-season roster data is empty")

    latest_week = rosters.get_column("week").max()
    return set(
        rosters.filter(
            (pl.col("week") == latest_week)
            & (pl.col("team") == "PIT")
            & pl.col("status").is_in(["ACT", "RES", "DEV"])
            & pl.col("gsis_id").is_not_null()
        )
        .get_column("gsis_id")
        .cast(pl.String)
        .to_list()
    )


def summarize_player_value(
    player_stats: pl.DataFrame,
    contracts: pl.DataFrame,
    seasons: list[int],
    current_steelers_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Aggregate player production and cap hits for the selected seasons.

    Offensive yards and touchdowns use passing plus rushing stats for QBs,
    rushing plus receiving stats for RBs, and receiving stats for WRs and TEs.
    Defensive takeaways count interceptions and opponent-fumble recoveries;
    forced fumbles are also reported, whether or not the defense recovered them.
    """
    required_stats = {
        "player_id",
        "player_display_name",
        "position_group",
        "season",
        "recent_team",
        "games",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "def_interceptions",
        "def_fumbles_forced",
        "fumble_recovery_opp",
        "def_tackles_solo",
        "def_tackle_assists",
        "def_sacks",
    }
    missing_stats = sorted(required_stats - set(player_stats.columns))
    if missing_stats:
        raise ValueError(f"player stats are missing required columns: {missing_stats}")
    if not {"gsis_id", "season_history"}.issubset(contracts.columns):
        raise ValueError("contracts are missing required player and season history columns")

    selected_seasons = {int(season) for season in seasons}
    if not selected_seasons:
        return pl.DataFrame(schema=PLAYER_VALUE_SCHEMA)

    cap_hits: dict[tuple[str, int, str], float] = {}
    for contract in contracts.select("gsis_id", "season_history").to_dicts():
        player_id = contract["gsis_id"]
        if not player_id:
            continue
        for season_entry in contract["season_history"] or []:
            try:
                season = int(season_entry.get("year"))
                cap_hit = float(season_entry.get("cap_number"))
            except (TypeError, ValueError):
                continue
            team = season_entry.get("team")
            if season not in selected_seasons or not team:
                continue
            key = (str(player_id), season, str(team))
            cap_hits[key] = max(cap_hits.get(key, cap_hit), cap_hit)

    cap_rows = [
        {
            "player_id": player_id,
            "season": season,
            "cap_hit_m": cap_hit,
        }
        for (player_id, season, team), cap_hit in cap_hits.items()
    ]
    if not cap_rows:
        return pl.DataFrame(schema=PLAYER_VALUE_SCHEMA)

    cap_by_player_season = pl.DataFrame(cap_rows).group_by(
        ["player_id", "season"]
    ).agg(pl.col("cap_hit_m").sum())

    stats = player_stats.filter(
        pl.col("season").is_in(sorted(selected_seasons))
        & pl.col("position_group").is_in(
            [*OFFENSIVE_POSITION_GROUPS, *DEFENSIVE_POSITION_GROUPS]
        )
        & pl.col("player_id").is_not_null()
    ).with_columns(
        pl.col("player_id").cast(pl.String),
        pl.col("player_display_name").fill_null("").alias("player_name"),
        pl.col("recent_team").fill_null("").alias("team"),
        pl.col("passing_yards").fill_null(0).cast(pl.Float64),
        pl.col("rushing_yards").fill_null(0).cast(pl.Float64),
        pl.col("receiving_yards").fill_null(0).cast(pl.Float64),
        pl.col("passing_tds").fill_null(0).cast(pl.Float64),
        pl.col("rushing_tds").fill_null(0).cast(pl.Float64),
        pl.col("receiving_tds").fill_null(0).cast(pl.Float64),
        pl.col("def_interceptions").fill_null(0).cast(pl.Float64),
        pl.col("def_fumbles_forced").fill_null(0).cast(pl.Float64),
        pl.col("fumble_recovery_opp").fill_null(0).cast(pl.Float64),
        pl.col("def_tackles_solo").fill_null(0).cast(pl.Float64),
        pl.col("def_tackle_assists").fill_null(0).cast(pl.Float64),
        pl.col("def_sacks").fill_null(0).cast(pl.Float64),
        pl.col("games").fill_null(0).cast(pl.Int64),
    ).with_columns(
        pl.when(pl.col("position_group") == "QB")
        .then(pl.col("passing_yards") + pl.col("rushing_yards"))
        .when(pl.col("position_group") == "RB")
        .then(pl.col("rushing_yards") + pl.col("receiving_yards"))
        .when(pl.col("position_group").is_in(["WR", "TE"]))
        .then(pl.col("receiving_yards"))
        .otherwise(0.0)
        .alias("offensive_yards"),
        pl.when(pl.col("position_group") == "QB")
        .then(pl.col("passing_tds") + pl.col("rushing_tds"))
        .when(pl.col("position_group") == "RB")
        .then(pl.col("rushing_tds") + pl.col("receiving_tds"))
        .when(pl.col("position_group").is_in(["WR", "TE"]))
        .then(pl.col("receiving_tds"))
        .otherwise(0.0)
        .alias("offensive_touchdowns"),
        (pl.col("def_interceptions") + pl.col("fumble_recovery_opp")).alias(
            "defensive_takeaways"
        ),
        (pl.col("def_tackles_solo") + pl.col("def_tackle_assists")).alias(
            "tackles"
        ),
    )

    by_player_season = stats.group_by(
        ["player_id", "season", "position_group"]
    ).agg(
        pl.col("player_name").filter(pl.col("player_name") != "").first(),
        pl.col("team").filter(pl.col("team") != "").unique().sort().str.join("/").alias("team"),
        pl.col("games").sum(),
        pl.col("offensive_yards").sum(),
        pl.col("offensive_touchdowns").sum(),
        pl.col("tackles").sum(),
        pl.col("def_sacks").sum().alias("sacks"),
        pl.col("def_interceptions").sum().alias("defensive_interceptions"),
        pl.col("def_fumbles_forced").sum().alias("forced_fumbles"),
        pl.col("fumble_recovery_opp").sum().alias("fumble_recoveries"),
        pl.col("defensive_takeaways").sum(),
    )
    valued = by_player_season.join(
        cap_by_player_season,
        on=["player_id", "season"],
        how="inner",
    ).filter(pl.col("cap_hit_m") > 0).with_columns(
        pl.col("player_id")
        .is_in(sorted(current_steelers_ids or set()))
        .alias("is_steelers")
    )
    if not valued.height:
        return pl.DataFrame(schema=PLAYER_VALUE_SCHEMA)

    return (
        valued.group_by(["player_id", "position_group"])
        .agg(
            pl.col("player_name").filter(pl.col("player_name") != "").first(),
            pl.col("team").filter(pl.col("team") != "").unique().sort().str.join(", "),
            pl.col("games").sum(),
            pl.col("season").n_unique().alias("seasons"),
            pl.col("cap_hit_m").sum(),
            pl.col("offensive_yards").sum(),
            pl.col("offensive_touchdowns").sum(),
            pl.col("tackles").sum(),
            pl.col("sacks").sum(),
            pl.col("defensive_interceptions").sum(),
            pl.col("forced_fumbles").sum(),
            pl.col("fumble_recoveries").sum(),
            pl.col("defensive_takeaways").sum(),
            pl.col("is_steelers").any(),
        )
        .select(list(PLAYER_VALUE_SCHEMA))
        .sort(["position_group", "player_name"])
    )