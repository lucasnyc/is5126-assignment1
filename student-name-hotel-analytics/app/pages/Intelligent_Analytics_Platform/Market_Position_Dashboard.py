import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path

# â”€â”€ DB connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "DB_PATH" in st.session_state:
    DB_PATH = st.session_state["DB_PATH"]
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DB_PATH = PROJECT_ROOT / "data" / "reviews_sample.db"

conn = sqlite3.connect(DB_PATH)

# â”€â”€ Page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.title("Market Position Dashboard")
st.caption(
    "Understand what affects hotel ratings - and where to invest to improve guest experience."
)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ASPECT_COLS = ["service", "cleanliness", "value", "location_rating", "sleep_quality", "rooms"]
ASPECT_LABELS = {
    "service": "Service",
    "cleanliness": "Cleanliness",
    "value": "Value",
    "location_rating": "Location",
    "sleep_quality": "Sleep Quality",
    "rooms": "Rooms",
}
ASPECT_COLORS = [
    "#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43", "#d73027"
]

# â”€â”€ Data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def load_reviews(_conn):
    df = pd.read_sql(
        """
        SELECT offering_id, overall, service, cleanliness, value, location_rating,
               sleep_quality, rooms, title, text, review_date
        FROM reviews
        WHERE overall BETWEEN 1 AND 5
          AND review_date IS NOT NULL
        ORDER BY review_date
        """,
        _conn,
    )
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["year_month"] = df["review_date"].dt.to_period("M")
    return df


@st.cache_data
def load_hotel_stats(_conn):
    return pd.read_sql(
        """
        SELECT offering_id,
               COUNT(*) AS n_reviews,
               AVG(overall)        AS avg_overall,
               AVG(service)        AS avg_service,
               AVG(cleanliness)    AS avg_cleanliness,
               AVG(value)          AS avg_value,
               AVG(location_rating)AS avg_location,
               AVG(sleep_quality)  AS avg_sleep_quality,
               AVG(rooms)          AS avg_rooms
        FROM reviews
        WHERE overall BETWEEN 1 AND 5
        GROUP BY offering_id
        HAVING COUNT(*) >= 50
        """,
        _conn,
    )


df = load_reviews(conn)
hotel_stats = load_hotel_stats(conn)


@st.cache_data
def load_reviews_with_offering(_conn):
    rdf = pd.read_sql(
        """
        SELECT offering_id, overall, review_date
        FROM reviews
        WHERE overall BETWEEN 1 AND 5
          AND review_date IS NOT NULL
        ORDER BY review_date
        """,
        _conn,
    )
    rdf["review_date"] = pd.to_datetime(rdf["review_date"], errors="coerce")
    return rdf.dropna(subset=["review_date"])


df_full = load_reviews_with_offering(conn)

# â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab1, tab2, tab3 = st.tabs(
    [
        "Overall Rating",
        "Position Among Top and Bottom Performers",
        "Contributor Activity Over Time",
    ]
)
with tab1:
    st.subheader("Overall Rating Trend")

    # -- Filters ----------------------------------------------------------------
    fc1, fc2 = st.columns([1, 2])

    with fc1:
        st.markdown("**Filter 1: Hotel ID / Offering ID**")
        hotel_id_t1 = st.text_input(
            "Offering ID",
            value="",
            placeholder="e.g. 93466  (leave blank for all)",
            key="hotel_id_t1",
        )
        if hotel_id_t1.strip():
            try:
                hid_check = int(hotel_id_t1.strip())
                match_t1 = df_full[df_full["offering_id"] == hid_check]
                if match_t1.empty:
                    st.warning(f"Offering ID {hid_check} not found in data.")
                else:
                    st.caption(f"{len(match_t1):,} total reviews for Hotel {hid_check}")
            except ValueError:
                st.error("Enter a valid numeric Offering ID.")

    with fc2:
        st.markdown("**Filter 2: Date Range**")
        all_periods_t1 = sorted(df_full["review_date"].dt.to_period("M").unique())
        period_labels_t1 = [str(p) for p in all_periods_t1]
        rc1, rc2 = st.columns(2)
        with rc1:
            start_t1 = st.selectbox(
                "From (Month-Year)",
                options=period_labels_t1,
                index=0,
                key="start_t1",
            )
        with rc2:
            end_t1 = st.selectbox(
                "To (Month-Year)",
                options=period_labels_t1,
                index=len(period_labels_t1) - 1,
                key="end_t1",
            )
        if start_t1 > end_t1:
            st.error("'From' date must be before or equal to 'To' date.")

    st.markdown("---")

    # -- Filter the data --------------------------------------------------------
    base_t1 = df_full.copy()
    base_t1["year_month"] = base_t1["review_date"].dt.to_period("M")
    start_p_t1 = pd.Period(start_t1, freq="M")
    end_p_t1   = pd.Period(end_t1,   freq="M")

    # All-hotels baseline (date range only)
    fdf_all_t1 = base_t1[
        (base_t1["year_month"] >= start_p_t1) &
        (base_t1["year_month"] <= end_p_t1)
    ]

    # Hotel-specific subset (date range + hotel filter)
    hid_t1 = None
    fdf_hotel_t1 = None
    if hotel_id_t1.strip():
        try:
            hid_t1 = int(hotel_id_t1.strip())
            fdf_hotel_t1 = fdf_all_t1[fdf_all_t1["offering_id"] == hid_t1]
            if fdf_hotel_t1.empty:
                fdf_hotel_t1 = None
        except ValueError:
            pass

    scope_t1 = f"Hotel {hotel_id_t1.strip()}" if hid_t1 else "All Hotels"

    if fdf_all_t1.empty:
        st.warning("No reviews match the selected date range.")
    else:
        gran_t1 = st.radio(
            "Time granularity",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True,
            key="gran_t1",
        )
        freq_t1 = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}[gran_t1]

        # Resample all-hotels average
        avg_all = (
            fdf_all_t1.set_index("review_date")["overall"]
            .resample(freq_t1).mean().reset_index()
            .rename(columns={"overall": "avg_all"})
        )

        # Resample hotel-specific average (only when a hotel is selected)
        avg_hotel = None
        if fdf_hotel_t1 is not None:
            avg_hotel = (
                fdf_hotel_t1.set_index("review_date")["overall"]
                .resample(freq_t1).mean().reset_index()
                .rename(columns={"overall": "avg_hotel"})
            )

        # Review volume for the selected scope
        vol_src = fdf_hotel_t1 if fdf_hotel_t1 is not None else fdf_all_t1
        vol_trend = (
            vol_src.set_index("review_date")["overall"]
            .resample(freq_t1).count().reset_index()
            .rename(columns={"overall": "n_reviews"})
        )

        fig_trend, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # All-hotels line (always shown)
        ax1.plot(
            avg_all["review_date"], avg_all["avg_all"],
            color="#aec7e8", linewidth=1.8, marker="o", markersize=2.5,
            label="All Hotels Avg",
        )

        # Selected-hotel line (shown only when a hotel ID is entered)
        if avg_hotel is not None:
            ax1.plot(
                avg_hotel["review_date"], avg_hotel["avg_hotel"],
                color="#d73027", linewidth=2.2, marker="o", markersize=3.5,
                label=f"Hotel {hid_t1} Avg",
            )

        ax1.set_ylabel("Avg Overall Rating")
        ax1.set_ylim(1, 5)
        title_sfx = f"Hotel {hid_t1} vs All Hotels" if hid_t1 else "All Hotels"
        ax1.set_title(f"Avg Overall Rating — {gran_t1} ({title_sfx})")
        ax1.legend(fontsize=9)

        ax2.fill_between(
            vol_trend["review_date"], vol_trend["n_reviews"],
            alpha=0.4, color="#74add1",
        )
        ax2.plot(
            vol_trend["review_date"], vol_trend["n_reviews"],
            color="#74add1", linewidth=1.5,
        )
        ax2.set_ylabel(f"Review Volume ({scope_t1})")

        fig_trend.tight_layout()
        st.pyplot(fig_trend)
        plt.close(fig_trend)

# =============================================================================
# Tab 3 -- Contributor Activity Over Time
# =============================================================================
with tab3:
    st.subheader("Contributor Activity Over Time")
    st.markdown(
        "Track how aspect scores — Service, Cleanliness, Value, Location, "
        "Sleep Quality, and Rooms — evolve over time. "
        "Enter a Hotel ID to overlay that hotel's trend against all hotels."
    )

    # -- Filters ---------------------------------------------------------------
    fc1_t3, fc2_t3 = st.columns([1, 2])
    with fc1_t3:
        st.markdown("**Filter 1: Hotel ID / Offering ID**")
        hotel_id_t3 = st.text_input(
            "Offering ID",
            value="",
            placeholder="e.g. 93466  (leave blank for all)",
            key="hotel_id_t3",
        )
        if hotel_id_t3.strip():
            try:
                hid_check_t3 = int(hotel_id_t3.strip())
                match_t3 = df[df["offering_id"] == hid_check_t3]
                if match_t3.empty:
                    st.warning(f"Offering ID {hid_check_t3} not found in data.")
                else:
                    st.caption(f"{len(match_t3):,} total reviews for Hotel {hid_check_t3}")
            except ValueError:
                st.error("Enter a valid numeric Offering ID.")

    with fc2_t3:
        st.markdown("**Filter 2: Date Range**")
        all_periods_t3 = sorted(df["year_month"].unique())
        period_labels_t3 = [str(p) for p in all_periods_t3]
        rc1_t3, rc2_t3 = st.columns(2)
        with rc1_t3:
            start_t3 = st.selectbox(
                "From (Month-Year)",
                options=period_labels_t3,
                index=0,
                key="start_t3",
            )
        with rc2_t3:
            end_t3 = st.selectbox(
                "To (Month-Year)",
                options=period_labels_t3,
                index=len(period_labels_t3) - 1,
                key="end_t3",
            )
        if start_t3 > end_t3:
            st.error("'From' date must be before or equal to 'To' date.")

    st.markdown("---")

    # -- Build filtered datasets -----------------------------------------------
    start_p_t3 = pd.Period(start_t3, freq="M")
    end_p_t3   = pd.Period(end_t3,   freq="M")

    df_all_t3 = df[
        (df["year_month"] >= start_p_t3) &
        (df["year_month"] <= end_p_t3)
    ]

    hid_t3 = None
    df_hotel_t3 = None
    if hotel_id_t3.strip():
        try:
            hid_t3 = int(hotel_id_t3.strip())
            candidate_t3 = df_all_t3[df_all_t3["offering_id"] == hid_t3]
            if not candidate_t3.empty:
                df_hotel_t3 = candidate_t3
        except ValueError:
            pass

    if df_all_t3.empty:
        st.warning("No reviews match the selected date range.")
    else:
        granularity_t3 = st.radio(
            "Time granularity",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True,
            key="granularity_t3",
        )
        freq_t3 = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}[granularity_t3]

        ts_all_t3 = (
            df_all_t3.set_index("review_date")[ASPECT_COLS]
            .resample(freq_t3).mean().reset_index()
        )
        ts_hotel_t3 = None
        if df_hotel_t3 is not None:
            ts_hotel_t3 = (
                df_hotel_t3.set_index("review_date")[ASPECT_COLS]
                .resample(freq_t3).mean().reset_index()
            )

        hotel_sfx_t3 = f" — Hotel {hid_t3} vs All Hotels" if hid_t3 else " — All Hotels"

        # Aspect Scores Over Time
        st.markdown("#### Aspect Scores Over Time")
        fig_ts3, ax = plt.subplots(figsize=(12, 5))
        for col, color in zip(ASPECT_COLS, ASPECT_COLORS):
            valid_all = ts_all_t3[["review_date", col]].dropna()
            if not valid_all.empty:
                ax.plot(
                    valid_all["review_date"], valid_all[col],
                    label=f"{ASPECT_LABELS[col]} (All)",
                    color=color, linewidth=1.4, linestyle="--",
                    alpha=0.55, marker="o", markersize=2,
                )
            if ts_hotel_t3 is not None:
                valid_h = ts_hotel_t3[["review_date", col]].dropna()
                if not valid_h.empty:
                    ax.plot(
                        valid_h["review_date"], valid_h[col],
                        label=f"{ASPECT_LABELS[col]} (Hotel {hid_t3})",
                        color=color, linewidth=2.2, linestyle="-",
                        marker="o", markersize=3.5,
                    )
        ax.set_ylabel("Average Rating (1-5)")
        ax.set_ylim(1, 5)
        ax.set_title(f"Aspect Ratings — {granularity_t3}{hotel_sfx_t3}")
        ax.legend(loc="lower left", fontsize=7, ncol=3)
        ax.axhline(3.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        fig_ts3.tight_layout()
        st.pyplot(fig_ts3)
        plt.close(fig_ts3)

        # Review Volume Over Time
        st.markdown("#### Review Volume Over Time")
        vol_src_t3 = df_hotel_t3 if df_hotel_t3 is not None else df_all_t3
        vol_t3 = (
            vol_src_t3.set_index("review_date")["overall"]
            .resample(freq_t3).count().reset_index()
            .rename(columns={"overall": "n_reviews"})
        )
        fig_vol3, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(vol_t3["review_date"], vol_t3["n_reviews"], alpha=0.4, color="#4575b4")
        ax.plot(vol_t3["review_date"], vol_t3["n_reviews"], color="#4575b4", linewidth=1.5)
        vol_label_t3 = f"Hotel {hid_t3}" if hid_t3 else "All Hotels"
        ax.set_ylabel("Number of Reviews")
        ax.set_title(f"Review Volume — {granularity_t3} ({vol_label_t3})")
        fig_vol3.tight_layout()
        st.pyplot(fig_vol3)
        plt.close(fig_vol3)

        # Individual Aspect Trend Charts
        with st.expander("Individual Aspect Trend Charts"):
            cols_grid_t3 = st.columns(3)
            for idx, (col, color) in enumerate(zip(ASPECT_COLS, ASPECT_COLORS)):
                fig_ind3, ax = plt.subplots(figsize=(5, 3))
                valid_all_ind = ts_all_t3[["review_date", col]].dropna()
                if not valid_all_ind.empty:
                    ax.plot(
                        valid_all_ind["review_date"], valid_all_ind[col],
                        color=color, linewidth=1.4, linestyle="--", alpha=0.55,
                        label="All Hotels",
                    )
                    ax.fill_between(
                        valid_all_ind["review_date"], valid_all_ind[col],
                        alpha=0.08, color=color,
                    )
                if ts_hotel_t3 is not None:
                    valid_h_ind = ts_hotel_t3[["review_date", col]].dropna()
                    if not valid_h_ind.empty:
                        ax.plot(
                            valid_h_ind["review_date"], valid_h_ind[col],
                            color=color, linewidth=2.2, linestyle="-",
                            label=f"Hotel {hid_t3}",
                        )
                        ax.fill_between(
                            valid_h_ind["review_date"], valid_h_ind[col],
                            alpha=0.18, color=color,
                        )
                ax.set_ylim(1, 5)
                ax.set_title(ASPECT_LABELS[col])
                ax.set_ylabel("Avg Rating")
                ax.legend(fontsize=7)
                fig_ind3.tight_layout()
                with cols_grid_t3[idx % 3]:
                    st.pyplot(fig_ind3)
                plt.close(fig_ind3)

        # Summary stats table
        st.markdown("#### Aspect Summary Statistics (Filtered Period)")
        stats_src_t3 = df_hotel_t3 if df_hotel_t3 is not None else df_all_t3
        stats_label_t3 = f"Hotel {hid_t3}" if hid_t3 else "All Hotels"
        stat_rows_t3 = []
        for col in ASPECT_COLS:
            s = stats_src_t3[col].dropna()
            stat_rows_t3.append({
                "Aspect": ASPECT_LABELS[col],
                f"Mean ({stats_label_t3})": round(s.mean(), 3),
                "Median": round(s.median(), 3),
                "Std Dev": round(s.std(), 3),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Reviews": int(s.count()),
            })
        st.dataframe(pd.DataFrame(stat_rows_t3), use_container_width=True, hide_index=True)

# =============================================================================
# Tab 2 -- Position Among Top and Bottom Performers
# =============================================================================
with tab2:
    st.subheader("Top & Bottom Performing Hotels")

    AVG_ASP = ["avg_service", "avg_cleanliness", "avg_value", "avg_location", "avg_sleep_quality", "avg_rooms"]
    DISPLAY  = ["Service", "Cleanliness", "Value", "Location", "Sleep Quality", "Rooms"]
    ASP_RAW  = ["service", "cleanliness", "value", "location_rating", "sleep_quality", "rooms"]

    # ── Filter row ────────────────────────────────────────────────────────────
    filter_col1, filter_col2 = st.columns([1, 2])

    with filter_col1:
        st.markdown("**Filter 1: Hotel ID / Offering ID**")
        my_hotel_id_input = st.text_input("Offering ID", value="", placeholder="e.g. 93466  (leave blank for all)", key="my_hotel_id")
        enable_my_hotel = False
        my_hotel_ratings = {}
        my_overall = None

        if my_hotel_id_input.strip():
            try:
                my_hotel_id = int(my_hotel_id_input.strip())
                my_row = hotel_stats[hotel_stats["offering_id"] == my_hotel_id]
                if my_row.empty:
                    st.warning(f"Offering ID {my_hotel_id} not found (needs >= 50 reviews).")
                else:
                    enable_my_hotel = True
                    r = my_row.iloc[0]
                    my_overall = r["avg_overall"]
                    my_hotel_ratings = {
                        "service":         r["avg_service"],
                        "cleanliness":     r["avg_cleanliness"],
                        "value":           r["avg_value"],
                        "location_rating": r["avg_location"],
                        "sleep_quality":   r["avg_sleep_quality"],
                        "rooms":           r["avg_rooms"],
                    }
                    st.caption(f"{int(r['n_reviews']):,} total reviews for Hotel {my_hotel_id}")
            except ValueError:
                st.error("Enter a valid numeric Offering ID.")

    with filter_col2:
        st.markdown("**Filter 2: Date Range**")
        all_periods_t2 = sorted(df["year_month"].unique())
        period_labels_t2 = [str(p) for p in all_periods_t2]
        rc1_t2, rc2_t2 = st.columns(2)
        with rc1_t2:
            start_t2 = st.selectbox(
                "From (Month-Year)",
                options=period_labels_t2,
                index=0,
                key="start_t2",
            )
        with rc2_t2:
            end_t2 = st.selectbox(
                "To (Month-Year)",
                options=period_labels_t2,
                index=len(period_labels_t2) - 1,
                key="end_t2",
            )
        if start_t2 > end_t2:
            st.error("'From' date must be before or equal to 'To' date.")

    st.markdown("---")

    # ── Recompute hotel stats filtered by date range ──────────────────────────
    @st.cache_data
    def hotel_stats_by_daterange(_conn, start_date: str, end_date: str):
        query = """
            SELECT offering_id,
                   COUNT(*) AS n_reviews,
                   AVG(overall)         AS avg_overall,
                   AVG(service)         AS avg_service,
                   AVG(cleanliness)     AS avg_cleanliness,
                   AVG(value)           AS avg_value,
                   AVG(location_rating) AS avg_location,
                   AVG(sleep_quality)   AS avg_sleep_quality,
                   AVG(rooms)           AS avg_rooms
            FROM reviews
            WHERE overall BETWEEN 1 AND 5
              AND review_date >= ?
              AND review_date <= ?
            GROUP BY offering_id
            HAVING COUNT(*) >= 10
        """
        import sqlite3 as _sqlite3
        _conn2 = _sqlite3.connect(str(DB_PATH))
        result = pd.read_sql(query, _conn2, params=[start_date, end_date])
        _conn2.close()
        return result

    start_d = pd.Period(start_t2, freq="M").start_time.date()
    end_d   = pd.Period(end_t2,   freq="M").end_time.date()

    hs = hotel_stats_by_daterange(conn, str(start_d), str(end_d))

    is_full_range = (start_t2 == period_labels_t2[0] and end_t2 == period_labels_t2[-1])
    label_sfx = " (all dates)" if is_full_range else f" ({start_t2} – {end_t2})"

    n = st.slider("Number of hotels to show", 5, 20, 10, key="top_n_slider")
    top_n = hs.nlargest(n, "avg_overall").reset_index(drop=True)
    bot_n = hs.nsmallest(n, "avg_overall").reset_index(drop=True)

    # ── Top / Bottom bar charts ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Performers**")
        fig5, ax = plt.subplots(figsize=(6, max(4, n * 0.45)))
        labels5 = top_n["offering_id"].astype(str).tolist()[::-1]
        vals5   = top_n["avg_overall"].tolist()[::-1]
        colors5 = ["#4575b4"] * len(labels5)
        if enable_my_hotel:
            labels5 = ["My Hotel"] + labels5
            vals5   = [my_overall] + vals5
            colors5 = ["#ff7f0e"] + colors5
        ax.barh(labels5, vals5, color=colors5)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Avg Overall Rating")
        ax.set_title(f"Top {n} Hotels{label_sfx}")
        for i, v in enumerate(vals5):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with c2:
        st.markdown("**Bottom Performers**")
        fig6, ax = plt.subplots(figsize=(6, max(4, n * 0.45)))
        labels6 = bot_n["offering_id"].astype(str).tolist()
        vals6   = bot_n["avg_overall"].tolist()
        colors6 = ["#d73027"] * len(labels6)
        if enable_my_hotel:
            labels6 = labels6 + ["Hotel ID / Offering ID"]
            vals6   = vals6   + [my_overall]
            colors6 = colors6 + ["#ff7f0e"]
        ax.barh(labels6, vals6, color=colors6)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Avg Overall Rating")
        ax.set_title(f"Bottom {n} Hotels{label_sfx}")
        for i, v in enumerate(vals6):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    # ── Aspect profile chart ──────────────────────────────────────────────────
    st.markdown("**Aspect Profile: Top vs Bottom Performers**" + label_sfx)
    top_mean = top_n[AVG_ASP].mean().values
    bot_mean = bot_n[AVG_ASP].mean().values
    x = np.arange(len(DISPLAY))
    width = 0.25 if enable_my_hotel else 0.35

    fig7, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width, top_mean, width, label="Top Performers", color="#4575b4")
    ax.bar(x,         bot_mean, width, label="Bottom Performers", color="#d73027")
    if enable_my_hotel:
        my_vals = [my_hotel_ratings.get(r, 0) for r in ASP_RAW]
        ax.bar(x + width, my_vals, width, label="My Hotel", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY, rotation=20, ha="right")
    ax.set_ylim(0, 5)
    ax.set_ylabel("Avg Rating")
    ax.set_title("Aspect Ratings Comparison")
    ax.legend()
    fig7.tight_layout()
    st.pyplot(fig7)
    plt.close(fig7)

    # ── My Hotel gap analysis scorecard ──────────────────────────────────────
    if enable_my_hotel:
        st.markdown("**My Hotel vs Top Performers — Gap Analysis**")
        rows = []
        for raw, label, avg_col in zip(ASP_RAW, DISPLAY, AVG_ASP):
            my_val  = my_hotel_ratings[raw]
            top_val = top_n[avg_col].mean()
            gap = my_val - top_val
            rows.append({
                "Aspect": label,
                "My Hotel": round(my_val, 2),
                "Top Performers Avg": round(top_val, 2),
                "Gap": round(gap, 2),
                "Status": "Above" if gap >= 0 else "Below",
            })
        gap_df = pd.DataFrame(rows)
        st.dataframe(
            gap_df.style.map(
                lambda v: "color: green" if v == "Above" else "color: red",
                subset=["Status"],
            ),
            use_container_width=True,
        )
