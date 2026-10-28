"""Local Parquet storage for the NFL source datasets used by the analyzer."""

from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
import polars as pl

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
FTN_FIRST_SEASON = 2022


def _season_file(dataset: str, season: int) -> Path:
    return DATA_DIR / f"{dataset}_{season}.parquet"


def _write_season(frame: pl.DataFrame, path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary_path)
    temporary_path.replace(path)


def _load_seasons(
    dataset: str,
    seasons: tuple[int, ...],
    loader: Callable[[int], pl.DataFrame],
) -> pl.DataFrame:
    frames = []
    for season in seasons:
        path = _season_file(dataset, season)
        if path.exists():
            frame = pl.read_parquet(path)
        else:
            frame = loader(season)
            _write_season(frame, path)
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed")


def load_play_by_play(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Read cached play-by-play files, downloading and saving missing seasons."""
    return _load_seasons(
        "play_by_play",
        seasons,
        lambda season: nfl.load_pbp(season),
    )


def load_ftn_charting(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Read cached FTN charting files, downloading missing eligible seasons."""
    unsupported = [season for season in seasons if season < FTN_FIRST_SEASON]
    if unsupported:
        raise ValueError(
            f"FTN charting is only available from {FTN_FIRST_SEASON}; "
            f"unsupported seasons: {unsupported}"
        )
    return _load_seasons(
        "ftn_charting",
        seasons,
        lambda season: nfl.load_ftn_charting(season),
    )


def download_default_datasets() -> dict[str, int]:
    """Save the app's default completed seasons and available FTN charting."""
    latest_completed_season = nfl.get_current_season() - 1
    seasons = tuple(range(max(1999, latest_completed_season - 4), latest_completed_season + 1))
    pbp = load_play_by_play(seasons)
    charting_seasons = tuple(
        season for season in seasons if season >= FTN_FIRST_SEASON
    )
    ftn = load_ftn_charting(charting_seasons)
    return {
        "play_by_play_rows": pbp.height,
        "ftn_charting_rows": ftn.height,
        "play_by_play_seasons": len(seasons),
        "ftn_charting_seasons": len(charting_seasons),
    }
