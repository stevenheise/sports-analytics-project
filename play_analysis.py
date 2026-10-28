"""Analysis helpers for Steelers plays, turnovers, and game outcomes."""

import polars as pl

SITUATIONS = [
    "1-5",
    "6-10",
    "11-20",
    "21-30",
    "31-40",
    "41-50",
    "50+",
]
PLAY_TYPES = [
    "Inside run",
    "Outside run",
    "Run (location unavailable)",
    "Pass - middle",
    "Pass - sideline",
    "Screen pass",
    "Pass (classification unavailable)",
]
TURNOVER_MARGIN_BANDS = [
    "Lost by 3+",
    "Lost by 2",
    "Lost by 1",
    "Even",
    "Won by 1",
    "Won by 2",
    "Won by 3+",
]
BLITZ_BANDS = [
    "No charted blitzers",
    "1 charted blitzer",
    "2+ charted blitzers",
    "Blitz count unavailable",
]


def summarize_plays(
    pbp: pl.DataFrame,
    seasons: list[int],
    down: int | None = None,
    ftn: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Summarize touchdown rates by play location and goal-line distance."""
    required_columns = {
        "posteam",
        "season",
        "down",
        "yardline_100",
        "play_type",
        "pass_touchdown",
        "rush_touchdown",
        "run_location",
        "pass_location",
        "game_id",
        "play_id",
    }
    missing_columns = sorted(required_columns - set(pbp.columns))
    if missing_columns:
        raise ValueError(
            "Play-by-play data is missing required columns: "
            + ", ".join(missing_columns)
        )

    goal_distance = pl.col("yardline_100")
    situation = (
        pl.when(goal_distance <= 5)
        .then(pl.lit("1-5"))
        .when(goal_distance <= 10)
        .then(pl.lit("6-10"))
        .when(goal_distance <= 20)
        .then(pl.lit("11-20"))
        .when(goal_distance <= 30)
        .then(pl.lit("21-30"))
        .when(goal_distance <= 40)
        .then(pl.lit("31-40"))
        .when(goal_distance <= 50)
        .then(pl.lit("41-50"))
        .otherwise(pl.lit("50+"))
    )
    plays = pbp.with_columns(
        pl.col("play_type").str.to_lowercase().alias("_play_type"),
        pl.col("play_id").cast(pl.Int64, strict=False).alias("_play_id"),
    )
    if ftn is None:
        plays = plays.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("_is_screen_pass")
        )
    else:
        ftn_required = {"nflverse_game_id", "nflverse_play_id", "is_screen_pass"}
        missing_ftn = sorted(ftn_required - set(ftn.columns))
        if missing_ftn:
            raise ValueError(
                "FTN charting data is missing required columns: "
                + ", ".join(missing_ftn)
            )
        ftn_screen = ftn.select(
            pl.col("nflverse_game_id").alias("_ftn_game_id"),
            pl.col("nflverse_play_id")
            .cast(pl.Int64, strict=False)
            .alias("_ftn_play_id"),
            pl.col("is_screen_pass").cast(pl.Boolean, strict=False),
        )
        duplicate_keys = (
            ftn_screen.group_by(["_ftn_game_id", "_ftn_play_id"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicate_keys.height:
            raise ValueError("FTN charting data contains duplicate game/play IDs")
        plays = plays.join(
            ftn_screen,
            left_on=["game_id", "_play_id"],
            right_on=["_ftn_game_id", "_ftn_play_id"],
            how="left",
        ).rename({"is_screen_pass": "_is_screen_pass"})

    plays = plays.filter(
        (pl.col("posteam") == "PIT")
        & pl.col("season").is_in(seasons)
        & pl.col("_play_type").is_in(["pass", "run"])
        & goal_distance.is_not_null()
        & (goal_distance > 0)
        & (goal_distance <= 100)
    )
    if down is not None:
        plays = plays.filter(pl.col("down") == down)

    play_category = (
        pl.when(pl.col("_play_type") == "run")
        .then(
            pl.when(pl.col("run_location") == "middle")
            .then(pl.lit("Inside run"))
            .when(pl.col("run_location").is_in(["left", "right"]))
            .then(pl.lit("Outside run"))
            .otherwise(pl.lit("Run (location unavailable)"))
        )
        .otherwise(
            pl.when(pl.col("_is_screen_pass") == True)
            .then(pl.lit("Screen pass"))
            .when(pl.col("_is_screen_pass").is_null())
            .then(pl.lit("Pass (classification unavailable)"))
            .when(pl.col("pass_location") == "middle")
            .then(pl.lit("Pass - middle"))
            .when(pl.col("pass_location").is_in(["left", "right"]))
            .then(pl.lit("Pass - sideline"))
            .otherwise(pl.lit("Pass (classification unavailable)"))
        )
    )
    touchdowns = (
        pl.when(pl.col("_play_type") == "pass")
        .then(pl.col("pass_touchdown").fill_null(0))
        .otherwise(pl.col("rush_touchdown").fill_null(0))
    )
    aggregates = (
        plays.with_columns(
            situation.alias("Situation"),
            play_category.alias("Play type"),
            touchdowns.alias("_touchdown"),
        )
        .group_by(["Situation", "Play type"])
        .agg(
            pl.len().alias("Plays"),
            pl.col("_touchdown").sum().cast(pl.Int64).alias("Touchdowns"),
            (pl.col("_touchdown").mean() * 100)
            .round(1)
            .alias("Touchdown rate (%)"),
        )
    )
    complete_rows = pl.DataFrame(
        {
            "Situation": [s for s in SITUATIONS for _ in PLAY_TYPES],
            "Play type": PLAY_TYPES * len(SITUATIONS),
        }
    )
    return (
        complete_rows.join(aggregates, on=["Situation", "Play type"], how="left")
        .with_columns(pl.col("Plays").fill_null(0).cast(pl.Int64))
        .sort(
            [
                pl.col("Situation").replace_strict(
                    {name: index for index, name in enumerate(SITUATIONS)}
                ),
                pl.col("Play type").replace_strict(
                    {name: index for index, name in enumerate(PLAY_TYPES)}
                ),
            ]
        )
    )


def summarize_turnover_margin_outcomes(
    pbp: pl.DataFrame,
    seasons: list[int],
) -> pl.DataFrame:
    """Summarize Steelers game outcomes by turnover margin."""
    required_columns = {
        "game_id",
        "season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "posteam",
        "play_type",
        "interception",
        "fumble_lost",
    }
    missing_columns = sorted(required_columns - set(pbp.columns))
    if missing_columns:
        raise ValueError(
            "Play-by-play data is missing required columns: "
            + ", ".join(missing_columns)
        )

    games = (
        pbp.filter(
            pl.col("season").is_in(seasons)
            & ((pl.col("home_team") == "PIT") | (pl.col("away_team") == "PIT"))
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        .select(
            "game_id",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        )
        .unique("game_id")
        .with_columns(
            pl.when(pl.col("home_team") == "PIT")
            .then(pl.col("home_score") - pl.col("away_score"))
            .otherwise(pl.col("away_score") - pl.col("home_score"))
            .alias("_score_margin")
        )
        .with_columns(
            pl.when(pl.col("_score_margin") > 0)
            .then(pl.lit("Win"))
            .when(pl.col("_score_margin") < 0)
            .then(pl.lit("Loss"))
            .otherwise(pl.lit("Tie"))
            .alias("Outcome")
        )
    )
    turnovers_by_game = (
        pbp.filter(
            pl.col("season").is_in(seasons)
            & pl.col("play_type").is_in(["pass", "run"])
        )
        .group_by(["game_id", "posteam"])
        .agg(
            (
                pl.col("interception").fill_null(0)
                + pl.col("fumble_lost").fill_null(0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("_turnovers")
        )
        .group_by("game_id")
        .agg(
            pl.when(pl.col("posteam") == "PIT")
            .then(pl.col("_turnovers"))
            .otherwise(0)
            .sum()
            .alias("_steelers_turnovers"),
            pl.when(pl.col("posteam") != "PIT")
            .then(pl.col("_turnovers"))
            .otherwise(0)
            .sum()
            .alias("_opponent_turnovers"),
        )
    )
    game_results = (
        games.join(turnovers_by_game, on="game_id", how="left")
        .with_columns(
            (
                pl.col("_opponent_turnovers").fill_null(0)
                - pl.col("_steelers_turnovers").fill_null(0)
            ).alias("_turnover_margin")
        )
        .with_columns(
            pl.when(pl.col("_turnover_margin") <= -3)
            .then(pl.lit("Lost by 3+"))
            .when(pl.col("_turnover_margin") == -2)
            .then(pl.lit("Lost by 2"))
            .when(pl.col("_turnover_margin") == -1)
            .then(pl.lit("Lost by 1"))
            .when(pl.col("_turnover_margin") == 0)
            .then(pl.lit("Even"))
            .when(pl.col("_turnover_margin") == 1)
            .then(pl.lit("Won by 1"))
            .when(pl.col("_turnover_margin") == 2)
            .then(pl.lit("Won by 2"))
            .otherwise(pl.lit("Won by 3+"))
            .alias("Turnover margin")
        )
    )
    aggregates = game_results.group_by("Turnover margin").agg(
        pl.len().alias("Games"),
        (pl.col("Outcome") == "Win").sum().alias("Wins"),
        (pl.col("Outcome") == "Loss").sum().alias("Losses"),
        (pl.col("Outcome") == "Tie").sum().alias("Ties"),
    )
    complete_rows = pl.DataFrame(
        {
            "Turnover margin": TURNOVER_MARGIN_BANDS,
            "_order": list(range(len(TURNOVER_MARGIN_BANDS))),
        }
    )
    return (
        complete_rows.join(aggregates, on="Turnover margin", how="left")
        .with_columns(
            pl.col("Games", "Wins", "Losses", "Ties").fill_null(0).cast(pl.Int64),
            pl.when(pl.col("Games") > 0)
            .then((pl.col("Wins") / pl.col("Games") * 100).round(1))
            .otherwise(None)
            .alias("Win rate (%)"),
        )
        .sort("_order")
        .drop("_order")
    )


def summarize_defensive_turnovers_by_blitz(
    pbp: pl.DataFrame,
    seasons: list[int],
    ftn: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Summarize Steelers defensive takeaways by charted pass-rush count."""
    required_columns = {
        "game_id",
        "play_id",
        "season",
        "posteam",
        "defteam",
        "play_type",
        "interception",
        "sack",
        "fumble_forced",
        "fumble_lost",
    }
    missing_columns = sorted(required_columns - set(pbp.columns))
    if missing_columns:
        raise ValueError(
            "Play-by-play data is missing required columns: "
            + ", ".join(missing_columns)
        )

    opponent_dropbacks = (
        pbp.filter(
            pl.col("season").is_in(seasons)
            & (pl.col("defteam") == "PIT")
            & (pl.col("posteam") != "PIT")
            & (pl.col("play_type") == "pass")
        )
        .with_columns(
            pl.col("play_id").cast(pl.Int64, strict=False).alias("_play_id")
        )
    )
    if ftn is None:
        opponent_dropbacks = opponent_dropbacks.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("_n_blitzers")
        )
    else:
        ftn_required = {"nflverse_game_id", "nflverse_play_id", "n_blitzers"}
        missing_ftn = sorted(ftn_required - set(ftn.columns))
        if missing_ftn:
            raise ValueError(
                "FTN charting data is missing required columns: "
                + ", ".join(missing_ftn)
            )
        ftn_blitzers = ftn.select(
            pl.col("nflverse_game_id").alias("_ftn_game_id"),
            pl.col("nflverse_play_id")
            .cast(pl.Int64, strict=False)
            .alias("_ftn_play_id"),
            pl.col("n_blitzers")
            .cast(pl.Int64, strict=False)
            .alias("_n_blitzers"),
        )
        duplicate_keys = (
            ftn_blitzers.group_by(["_ftn_game_id", "_ftn_play_id"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicate_keys.height:
            raise ValueError("FTN charting data contains duplicate game/play IDs")
        opponent_dropbacks = opponent_dropbacks.join(
            ftn_blitzers,
            left_on=["game_id", "_play_id"],
            right_on=["_ftn_game_id", "_ftn_play_id"],
            how="left",
        )

    charted_results = (
        opponent_dropbacks.with_columns(
            pl.when(pl.col("_n_blitzers") == 0)
            .then(pl.lit("No charted blitzers"))
            .when(pl.col("_n_blitzers") == 1)
            .then(pl.lit("1 charted blitzer"))
            .when(pl.col("_n_blitzers") >= 2)
            .then(pl.lit("2+ charted blitzers"))
            .otherwise(pl.lit("Blitz count unavailable"))
            .alias("Blitz pressure"),
            (
                pl.col("sack").fill_null(0)
                * pl.col("fumble_forced").fill_null(0)
            ).alias("_strip_sack"),
            (
                pl.col("sack").fill_null(0)
                * pl.col("fumble_lost").fill_null(0)
            ).alias("_recovered_strip_sack"),
        )
        .group_by("Blitz pressure")
        .agg(
            pl.len().alias("Pass plays"),
            pl.col("interception")
            .fill_null(0)
            .sum()
            .cast(pl.Int64)
            .alias("Interceptions"),
            pl.col("_strip_sack")
            .sum()
            .cast(pl.Int64)
            .alias("Strip sacks (forced)"),
            pl.col("_recovered_strip_sack")
            .sum()
            .cast(pl.Int64)
            .alias("Strip sacks recovered"),
        )
    )
    complete_rows = pl.DataFrame(
        {
            "Blitz pressure": BLITZ_BANDS,
            "_order": list(range(len(BLITZ_BANDS))),
        }
    )
    return (
        complete_rows.join(charted_results, on="Blitz pressure", how="left")
        .with_columns(
            pl.col(
                "Pass plays",
                "Interceptions",
                "Strip sacks (forced)",
                "Strip sacks recovered",
            )
            .fill_null(0)
            .cast(pl.Int64),
            pl.when(pl.col("Pass plays") > 0)
            .then(
                (pl.col("Interceptions") / pl.col("Pass plays") * 100).round(1)
            )
            .otherwise(None)
            .alias("Interception rate (%)"),
            pl.when(pl.col("Pass plays") > 0)
            .then(
                (
                    pl.col("Strip sacks (forced)")
                    / pl.col("Pass plays")
                    * 100
                ).round(1)
            )
            .otherwise(None)
            .alias("Strip-sack rate (%)"),
        )
        .sort("_order")
        .drop("_order")
    )
