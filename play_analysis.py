"""Analysis helpers for Steelers touchdown rates by field position."""

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
