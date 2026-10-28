"""Streamlit dashboard for Pittsburgh Steelers play and game analysis."""

from datetime import date

import nflreadpy as nfl
import mysql.connector
import polars as pl
import streamlit as st

from database import DatabaseConfigurationError, save_analysis_summaries
from nfl_data import DATA_DIR, load_ftn_charting as read_ftn_charting
from nfl_data import load_play_by_play as read_play_by_play
from play_analysis import (
    summarize_defensive_turnovers_by_blitz,
    summarize_plays,
    summarize_turnover_margin_outcomes,
)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_play_by_play(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Read local play-by-play files, downloading and caching missing seasons."""
    return read_play_by_play(seasons)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_ftn_charting(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Read local FTN files, downloading and caching missing seasons."""
    return read_ftn_charting(seasons)


def main() -> None:
    st.set_page_config(
        page_title="Steelers Touchdown Rate Analyzer",
        page_icon="🏈",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root { --steelers-gold: #ffb612; --steelers-black: #101010; }
        .stApp { background-color: #f5f5f3; }
        section[data-testid="stMain"] {
            background-color: #f5f5f3;
            color: #101010;
        }
        [data-testid="stHeader"] { background-color: rgba(245, 245, 243, 0.9); }
        section[data-testid="stMain"] h1,
        section[data-testid="stMain"] h2,
        section[data-testid="stMain"] h3,
        section[data-testid="stMain"] h4,
        section[data-testid="stMain"] p,
        section[data-testid="stMain"] label {
            color: var(--steelers-black) !important;
        }
        section[data-testid="stMain"] table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
        }
        section[data-testid="stMain"] th {
            background: #ffb612;
            color: #101010 !important;
            text-align: left;
        }
        section[data-testid="stMain"] th,
        section[data-testid="stMain"] td {
            padding: .55rem .7rem;
            border-bottom: 1px solid #d8d8d8;
        }
        [data-testid="stSidebar"] { background-color: #101010; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #ffffff;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: #ffb612; color: #101010;
        }
        div[data-testid="stMetric"] {
            background: white; border-left: 4px solid var(--steelers-gold);
            padding: .8rem; border-radius: .35rem;
        }
        .stApp section[data-testid="stMain"] [data-testid="stMetric"] * {
            color: #101010 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='color:#101010; border-bottom:4px solid #ffb612;"
        " padding-bottom:.35rem'>Steelers Touchdown Rate Analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Compare Pittsburgh's pass and rush touchdown rates by **yards from the "
        "opponent's goal line**. Each rate is touchdowns divided by qualifying "
        "scrimmage plays from that distance."
    )

    latest_completed_season = nfl.get_current_season() - 1
    available_seasons = list(range(1999, latest_completed_season + 2))
    default_seasons = list(
        range(max(1999, latest_completed_season - 4), latest_completed_season + 1)
    )
    with st.sidebar:
        st.header("Filters")
        selected_seasons = st.multiselect(
            "Season(s)",
            options=available_seasons,
            default=default_seasons,
            format_func=str,
            help="The default is the five most recently completed seasons.",
        )
        selected_down = st.selectbox(
            "Down",
            options=[None, 1, 2, 3, 4],
            format_func=lambda value: "All downs" if value is None else f"{value}st"
            if value == 1
            else f"{value}nd"
            if value == 2
            else f"{value}rd"
            if value == 3
            else f"{value}th",
        )

    if not selected_seasons:
        st.info("Choose at least one season in the sidebar to load play-by-play data.")
        return

    season_key = tuple(sorted(selected_seasons))
    with st.spinner(f"Loading nflverse play-by-play for {len(season_key)} season(s)…"):
        pbp = load_play_by_play(season_key)
    charting_seasons = tuple(season for season in season_key if season >= 2022)
    ftn = None
    if charting_seasons:
        with st.spinner("Loading FTN charting labels for screen passes…"):
            ftn = load_ftn_charting(charting_seasons)
    summary = summarize_plays(
        pbp, list(season_key), down=selected_down, ftn=ftn
    )
    turnover_outcomes = summarize_turnover_margin_outcomes(
        pbp, list(season_key)
    )
    defensive_turnovers = summarize_defensive_turnovers_by_blitz(
        pbp, list(season_key), ftn=ftn
    )
    play_count = int(summary.get_column("Plays").sum())

    selected_down_label = "all downs" if selected_down is None else f"down {selected_down}"
    st.caption(
        f"Seasons: {', '.join(map(str, season_key))} · "
        f"Filter: {selected_down_label} · Loaded {date.today().isoformat()} · "
        f"Local source cache: `{DATA_DIR.relative_to(DATA_DIR.parent.parent)}`"
    )
    total_col, scope_col = st.columns(2)
    total_col.metric("Qualifying plays", f"{play_count:,}")
    scope_col.metric("Analysis groups", "Run / pass splits")

    if play_count == 0:
        st.warning("No qualifying Steelers pass or run plays match these filters.")
    else:
        present_rates = summary.filter(pl.col("Plays") > 0)
        leaders = (
            present_rates.sort(
                ["Touchdown rate (%)", "Plays"],
                descending=[True, True],
            )
            .head(1)
        )
        if leaders.height:
            leader = leaders.row(0, named=True)
            st.info(
                f"Highest observed touchdown rate in this filtered sample: "
                f"**{leader['Play type']}** plays with **{leader['Situation']}** "
                f"yards from the opponent's goal line "
                f"({leader['Touchdown rate (%)']}% across "
                f"{leader['Plays']:,} plays). This is descriptive, not a "
                "guarantee of future results."
            )

    st.subheader("Touchdown rate by distance to the goal line")
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Touchdown rate (%)": st.column_config.NumberColumn(
                "Touchdown rate",
                format="%.1f%%",
            )
        },
    )

    st.subheader("Win rate by turnover margin")
    st.caption(
        "Turnover margin is Steelers takeaways minus giveaways. Win rate is "
        "wins divided by games, with ties counted as non-wins."
    )
    st.dataframe(
        turnover_outcomes,
        hide_index=True,
        width="stretch",
        column_config={
            "Win rate (%)": st.column_config.NumberColumn(
                "Win rate",
                format="%.1f%%",
            )
        },
    )

    st.subheader("Steelers defensive takeaways by blitz count")
    st.caption(
        "Interceptions and sack-fumbles forced on opposing pass plays. Rates "
        "use pass plays in each group as the denominator. FTN charting does not "
        "identify coverage type, blitzer position, or rush direction."
    )
    st.dataframe(
        defensive_turnovers,
        hide_index=True,
        width="stretch",
        column_config={
            "Interception rate (%)": st.column_config.NumberColumn(
                "Interception rate",
                format="%.1f%%",
            ),
            "Strip-sack rate (%)": st.column_config.NumberColumn(
                "Strip-sack rate",
                format="%.1f%%",
            ),
        },
    )

    st.subheader("Save dashboard summaries to TiDB")
    st.caption(
        "Saves the touchdown, turnover-margin, and defensive blitz summaries "
        "for the selected seasons to their respective TiDB tables. The "
        "touchdown summary also includes the selected down filter. Raw "
        "play-by-play is not stored."
    )
    if st.button("Save all summaries to TiDB", disabled=play_count == 0):
        try:
            with st.spinner("Saving dashboard summaries to TiDB…"):
                run_id = save_analysis_summaries(
                    summary,
                    turnover_outcomes,
                    defensive_turnovers,
                    list(season_key),
                    selected_down,
                )
        except DatabaseConfigurationError as exc:
            st.error(str(exc))
        except mysql.connector.Error as exc:
            if exc.errno in {1045, 1105}:
                st.error(
                    f"TiDB rejected the login (error {exc.errno}). Verify the "
                    "exact cluster-prefixed database username and password "
                    "shown by TiDB Cloud's Connect panel in your local `.env` file."
                )
            else:
                st.error(
                    "TiDB could not save the summary "
                    f"(MySQL error {exc.errno}). Check the connection settings, "
                    "TLS/network access, and database permissions."
                )
        else:
            st.success(
                f"All dashboard summaries saved to TiDB. Run ID: `{run_id}`"
            )

    st.subheader("How to interpret this")
    st.markdown(
        """
        - **Distance** is nflverse's `yardline_100`: yards from the opponent's goal
          line at the start of the play. These ranges are not yards-to-go for a first
          down. **50+** means more than 50 yards from the opponent's goal line;
          exactly 50 is in **41–50**.
          - **Inside run** means nflverse `run_location` is `middle`. **Outside run**
            means it is `left` or `right`. Runs with no location are shown separately.
          - **Pass - middle** uses nflverse `pass_location=middle`; **Pass - sideline**
            uses `left` or `right`. **Screen pass** uses the FTN charting
            `is_screen_pass` flag, joined by game and play ID.
          - **Unclassified** rows retain plays whose location or screen classification
            is missing; they are not silently assigned to another category.
          - **Plays** counts Steelers pass or run plays that began in that distance
            band. **Touchdowns** counts passing or rushing touchdowns on those plays.
          - **Touchdown rate** is `Touchdowns / Plays * 100` for each play type and
          distance band. It is a per-play rate, not the share of all team touchdowns.

          These are historical descriptions of Pittsburgh's offense, not causal
          estimates or play-calling recommendations. Small play counts can make
          rates volatile; opponent, score, clock, field position, and personnel are
          not controlled for. Plays missing field position and penalty/no-play and
          special-teams plays are excluded. FTN charting is available from 2022;
          for earlier seasons, pass type cannot be reliably identified and those
          plays appear as **Pass (classification unavailable)**.
          """
    )
    st.caption(
          "Source: NFL play-by-play data from nflverse, loaded with nflreadpy. "
          "Touchdowns use `pass_touchdown` and `rush_touchdown`; screen labels use "
          "the separate FTN charting dataset. "
        "The source may be updated over time; results reflect the data currently available."
    )


if __name__ == "__main__":
    main()
