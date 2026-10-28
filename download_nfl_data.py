"""Download the app's default NFL source data into ./data/raw."""

from nfl_data import DATA_DIR, download_default_datasets


if __name__ == "__main__":
    counts = download_default_datasets()
    print(f"Saved NFL source data under: {DATA_DIR}")
    print(
        f"Play-by-play: {counts['play_by_play_rows']:,} rows across "
        f"{counts['play_by_play_seasons']} seasons"
    )
    print(
        f"FTN charting: {counts['ftn_charting_rows']:,} rows across "
        f"{counts['ftn_charting_seasons']} seasons"
    )
