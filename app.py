"""Streamlit dashboard for Pittsburgh Steelers play and game analysis."""

from datetime import date

import nflreadpy as nfl
import mysql.connector
import polars as pl
import streamlit as st

from database import DatabaseConfigurationError, save_analysis_summaries
from nfl_data import DATA_DIR, load_ftn_charting as read_ftn_charting
from nfl_data import load_play_by_play as read_play_by_play
from player_analysis import (
    DEFENSIVE_POSITION_GROUPS,
    OFFENSIVE_POSITION_GROUPS,
    current_active_player_ids,
    current_steelers_roster_ids,
    rank_player_values,
    summarize_player_value,
)
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


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_player_stats(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Load and cache regular-season leaguewide player statistics."""
    return nfl.load_player_stats(list(seasons), summary_level="reg")


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_contracts() -> pl.DataFrame:
    """Load and cache historical player contract cap hits."""
    return nfl.load_contracts()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_current_rosters() -> pl.DataFrame:
    """Load current-season roster snapshots for active player identification."""
    return nfl.load_rosters(nfl.get_current_season())


def _render_player_leaderboard(
    player_values: pl.DataFrame,
    side: str,
    position_groups: tuple[str, ...],
) -> None:
    if side == "offense":
        metric_options = {
            "Yards": "offensive_yards",
            "Touchdowns (QB: TD/INT ratio)": "offensive_touchdowns",
        }
    else:
        metric_options = {
            "Tackles": "tackles",
            "Sacks": "sacks",
            "Takeaways": "defensive_takeaways",
        }

    controls = st.columns(3)
    position_filter = controls[0].selectbox(
        "Position group",
        options=["All", *position_groups],
        key=f"{side}_position_group",
    )
    metric_label = controls[1].selectbox(
        "Rank by",
        options=list(metric_options),
        key=f"{side}_metric",
    )
    metric_column = metric_options[metric_label]
    if side == "offense":
        value_label = (
            "Yards" if metric_column == "offensive_yards" else "TDs / TD-INT"
        )
    else:
        value_label = metric_label
    minimum_games = controls[2].selectbox(
        "Minimum games",
        options=[0, 8, 16, 32, 64],
        index=2,
        key=f"{side}_minimum_games",
    )
    scope = st.selectbox(
        "Player scope",
        options=[
            "Entire league",
            "Currently active players",
            "Steelers players only",
        ],
        key=f"{side}_scope",
    )

    ranked = rank_player_values(
        player_values,
        position_groups,
        metric_column,
        minimum_games,
        scope,
        position_filter,
    )

    if ranked.is_empty():
        st.info("No players with matching production and cap-hit data for these filters.")
        return

    st.caption(
        f"Top five per group · {metric_label}, ranked by production / ln(cap hit in dollars), "
        "within each position group. "
        "Steelers-only scope includes players on the latest listed Steelers roster. Active-player "
        "scope uses players listed as active on the latest roster week."
    )
    chart_rows = ranked.filter(pl.col("Position rank") <= 5).with_columns(
        pl.concat_str(
            ["position_group", pl.lit(" · "), "player_name"]
        ).alias("Player")
    )
    st.bar_chart(
        chart_rows.select("Player", "Rank score"),
        x="Player",
        y="Rank score",
        x_label="Player",
        y_label="Production / ln(cap hit in $)",
    )
    display = ranked.select(
        "Position rank",
        "player_name",
        "team",
        "position_group",
        "games",
        "seasons",
        "cap_hit_m",
        "Rank production",
        "Rank score",
        "is_steelers",
    ).rename(
        {
            "player_name": "Player",
            "team": "Team(s)",
            "position_group": "Position",
            "games": "Games",
            "seasons": "Seasons",
            "cap_hit_m": "Cap hit ($M)",
            "Rank production": value_label,
            "is_steelers": "Steelers player",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Cap hit ($M)": st.column_config.NumberColumn(format="$%.2f"),
            value_label: st.column_config.NumberColumn(format="%.2f"),
            "Rank score": st.column_config.NumberColumn(
                format="%.4f",
                help="Selected production divided by the natural log of cap hit in dollars.",
            ),
            "Steelers player": st.column_config.CheckboxColumn(),
        },
    )


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
        " padding-bottom:.35rem'>Steelers Analytics</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Explore Pittsburgh's scoring, player production, turnover trends, and "
        "defensive takeaways across the selected seasons."
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
        page = st.radio(
            "Dashboard page",
            [
                "Overview",
                "Player Value",
                "Touchdown Rate",
                "Turnover Margin",
                "Defense by Blitz",
            ],
            key="dashboard_page",
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
    with st.spinner("Loading leaguewide player stats and contract cap hits…"):
        player_stats = load_player_stats(season_key)
        contracts = load_contracts()
        current_rosters = load_current_rosters()
    current_active_ids = current_active_player_ids(current_rosters)
    current_steelers_ids = current_steelers_roster_ids(current_rosters)
    player_values = summarize_player_value(
        player_stats,
        contracts,
        list(season_key),
        current_steelers_ids=current_steelers_ids,
        current_active_ids=current_active_ids,
    )
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
    if page == "Overview":
        st.subheader("Your selected analysis")
        total_col, scope_col = st.columns(2)
        total_col.metric("Qualifying Steelers plays", f"{play_count:,}")
        scope_col.metric("Seasons selected", len(season_key))
        st.write(
            "Use the Dashboard page menu in the sidebar to explore player value, "
            "touchdown rates, turnover margins, or defensive takeaways separately."
        )
        st.caption(
            "Player rankings use production divided by the natural log of cap hit "
            "in dollars. Quarterbacks use a smoothed passing TD/INT ratio."
        )
        st.subheader("Save dashboard summaries to TiDB")
        st.caption(
            "Saves the touchdown, turnover-margin, defensive blitz, and player "
            "summaries for the selected seasons. Raw play-by-play is not stored."
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
                        player_values,
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
    elif page == "Player Value":
        st.subheader("Leaguewide player production and cap efficiency")
        st.caption(
            "Rank score is selected production divided by ln(cap hit in dollars). "
            "Quarterbacks use (passing TDs + 1) / (passing interceptions + 1); "
            "other players use role-based yards or touchdowns. Defensive rankings "
            "also include tackles, sacks, and takeaways. Rankings are within "
            "position group. The Steelers-only filter uses the latest roster."
        )
        offense_tab, defense_tab = st.tabs(["Offense", "Defense"])
        with offense_tab:
            _render_player_leaderboard(
                player_values,
                "offense",
                OFFENSIVE_POSITION_GROUPS,
            )
        with defense_tab:
            _render_player_leaderboard(
                player_values,
                "defense",
                DEFENSIVE_POSITION_GROUPS,
            )
    elif page == "Touchdown Rate":
        st.subheader("Touchdown rate by distance to the goal line")
        if play_count == 0:
            st.warning("No qualifying Steelers pass or run plays match these filters.")
        else:
            present_rates = summary.filter(pl.col("Plays") > 0)
            leaders = present_rates.sort(
                ["Touchdown rate (%)", "Plays"],
                descending=[True, True],
            ).head(1)
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
        st.subheader("How to interpret this")
        st.markdown(
            """
            Distance is nflverse's `yardline_100`: yards from the opponent's goal
            line at the start of the play. Inside/outside runs and pass locations
            come from nflverse; screen labels come from FTN charting. Missing
            classification is kept unclassified rather than guessed. Touchdown
            rate is touchdowns divided by qualifying pass/run plays in that band.
            These are historical descriptions, not causal estimates.
            """
        )
        st.caption(
            "Source: nflverse play-by-play and FTN charting. Results reflect the "
            "data currently available."
        )
    elif page == "Turnover Margin":
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
    elif page == "Defense by Blitz":
        st.subheader("Steelers defensive takeaways by blitz count")
        st.caption(
            "Interceptions and sack-fumbles forced on opposing pass plays. Rates "
            "use pass plays in each group as the denominator. FTN charting does "
            "not identify coverage type, blitzer position, or rush direction."
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



if __name__ == "__main__":
    main()
