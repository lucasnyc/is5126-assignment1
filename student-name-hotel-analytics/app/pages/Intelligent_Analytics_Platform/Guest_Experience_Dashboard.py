import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path

# -- DB connection ------------------------------------------------------------
if "DB_PATH" in st.session_state:
    DB_PATH = Path(st.session_state["DB_PATH"])
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DB_PATH = PROJECT_ROOT / "data" / "reviews_sample.db"

conn = sqlite3.connect(str(DB_PATH))

# -- Page header --------------------------------------------------------------
st.title("Guest Experience Dashboard")
st.caption(
    "Performance snapshot per hotel, aspect activity trends over time, "
    "and holistic text insights from guest reviews."
)

# -- Constants ----------------------------------------------------------------
ASPECT_COLS = [
    "service", "cleanliness", "value",
    "location_rating", "sleep_quality", "rooms",
]
ASPECT_LABELS = {
    "service":         "Service",
    "cleanliness":     "Cleanliness",
    "value":           "Value",
    "location_rating": "Location",
    "sleep_quality":   "Sleep Quality",
    "rooms":           "Rooms",
}
ASPECT_COLORS = [
    "#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43", "#d73027"
]

# -- Data loaders -------------------------------------------------------------
@st.cache_data
def load_reviews(_conn):
    df = pd.read_sql(
        """
        SELECT overall, service, cleanliness, value, location_rating,
               sleep_quality, rooms, title, text, review_date, offering_id
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


df = load_reviews(conn)


@st.cache_data
def load_reviews_with_authors(_conn):
    rdf = pd.read_sql(
        """
        SELECT r.offering_id, r.overall, r.service, r.cleanliness, r.value,
               r.location_rating, r.sleep_quality, r.rooms,
               r.title, r.text, r.review_date,
               a.author_name, a.author_location,
               a.author_num_reviews, a.author_num_helpful_votes
        FROM reviews r
        LEFT JOIN authors a ON r.author_id = a.author_id
        WHERE r.overall BETWEEN 1 AND 5
          AND r.review_date IS NOT NULL
        ORDER BY r.review_date
        """,
        _conn,
    )
    rdf["review_date"] = pd.to_datetime(rdf["review_date"], errors="coerce")
    rdf = rdf.dropna(subset=["review_date"])
    rdf["year_month"] = rdf["review_date"].dt.to_period("M")
    return rdf


df_rich = load_reviews_with_authors(conn)

# Build sorted list of year-month periods for the filter dropdowns
all_periods = sorted(df["year_month"].unique())
period_labels = [str(p) for p in all_periods]   # e.g. ["2010-01", ..., "2017-12"]


def filter_df(base_df, hotel_id_str: str, start_label: str, end_label: str):
    """Return a filtered copy of base_df given raw user inputs."""
    out = base_df.copy()
    start = pd.Period(start_label, freq="M")
    end   = pd.Period(end_label,   freq="M")
    out = out[(out["year_month"] >= start) & (out["year_month"] <= end)]
    if hotel_id_str.strip():
        try:
            hid = int(hotel_id_str.strip())
            out = out[out["offering_id"] == hid]
        except ValueError:
            pass
    return out


def draw_filters(tab_key: str):
    """Render the two shared filters; return (hotel_id_str, start_label, end_label)."""
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        st.markdown("**Filter: Offering ID (Hotel)**")
        hotel_id_str = st.text_input(
            "Offering ID",
            value="",
            placeholder="e.g. 93466  (leave blank for all)",
            key=f"hotel_id_{tab_key}",
        )
        if hotel_id_str.strip():
            try:
                hid = int(hotel_id_str.strip())
                match = df[df["offering_id"] == hid]
                if match.empty:
                    st.warning(f"Offering ID {hid} not found in data.")
                else:
                    st.caption(f"{len(match):,} reviews for hotel {hid}")
            except ValueError:
                st.error("Enter a valid numeric Offering ID.")

    with fc2:
        st.markdown("**Filter: Month-Year Date Range**")
        rc1, rc2 = st.columns(2)
        with rc1:
            start_label = st.selectbox(
                "From (Month-Year)",
                options=period_labels,
                index=0,
                key=f"start_period_{tab_key}",
            )
        with rc2:
            end_label = st.selectbox(
                "To (Month-Year)",
                options=period_labels,
                index=len(period_labels) - 1,
                key=f"end_period_{tab_key}",
            )
        if start_label > end_label:
            st.error("'From' date must be before or equal to 'To' date.")

    return hotel_id_str, start_label, end_label


# =============================================================================
# Global Filters (applied to all tabs)
# =============================================================================
hotel_id_global, start_global, end_global = draw_filters("global")
st.markdown("---")

scope_global = (
    f"Hotel {hotel_id_global.strip()}" if hotel_id_global.strip() else "All Hotels"
)
fdf_global = filter_df(df_rich, hotel_id_global, start_global, end_global)
fdf_base   = filter_df(df,      hotel_id_global, start_global, end_global)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overall Rating Distribution",
    "Top and Bottom Reviews",
    "Contributor Activity Over Time",
    "Textual Feedback Analytics",
    "Top Reviewer Details",
])

# =============================================================================
# Tab 2 -- Top and Bottom Reviews
# =============================================================================
with tab2:
    st.subheader("Top and Bottom Reviews")

    fdf = fdf_global

    if fdf.empty:
        st.warning("No reviews match the selected filters.")
    else:
        n_rows = st.slider("Reviews to display per group", 5, 30, 10, key="n_rows_t1")
        cols_show = ["offering_id", "review_date", "overall"] + ASPECT_COLS + ["title"]
        rename_map = {"offering_id": "Hotel ID", "review_date": "Date",
                      "overall": "Overall", "title": "Title", **ASPECT_LABELS}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Reviews** (highest overall rating)")
            top_rev = (
                fdf.nlargest(n_rows, "overall")[cols_show]
                .rename(columns=rename_map)
                .reset_index(drop=True)
            )
            st.dataframe(top_rev, use_container_width=True)

        with c2:
            st.markdown("**Bottom Reviews** (lowest overall rating)")
            bot_rev = (
                fdf.nsmallest(n_rows, "overall")[cols_show]
                .rename(columns=rename_map)
                .reset_index(drop=True)
            )
            st.dataframe(bot_rev, use_container_width=True)

        with st.expander("Read Full Review Texts"):
            for _, row in fdf.nlargest(n_rows, "overall").iterrows():
                loc_str = row["author_location"] if pd.notna(row["author_location"]) else "Unknown location"
                name_str = row["author_name"] if pd.notna(row["author_name"]) else "Anonymous"
                num_rev = int(row["author_num_reviews"]) if pd.notna(row["author_num_reviews"]) else "?"
                st.markdown(
                    f"**{row['title']}** — Overall: {row['overall']:.0f}/5 "
                    f"| Hotel {int(row['offering_id'])} "
                    f"| {str(row['review_date'])[:10]}  \n"
                    f"*{name_str}* from **{loc_str}** ({num_rev} reviews total)"
                )
                if pd.notna(row.get("text")):
                    st.write(row["text"])
                st.markdown("---")

# =============================================================================
# Tab 3 -- Contributor Activity Over Time
# =============================================================================
with tab3:
    st.subheader("Contributor Activity Over Time")
    st.markdown(
        "Track how aspect scores -- Service, Cleanliness, Value, Location, "
        "Sleep Quality, and Rooms -- evolve over time for a specific hotel or "
        "across all hotels."
    )

    fdf2 = fdf_base

    if fdf2.empty:
        st.warning("No reviews match the selected filters.")
    else:
        granularity = st.radio(
            "Time granularity",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True,
            key="granularity_t2",
        )
        freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        freq = freq_map[granularity]

        ts = (
            fdf2.set_index("review_date")[ASPECT_COLS]
            .resample(freq)
            .mean()
            .reset_index()
        )

        # Aspect trend lines
        st.markdown("#### Aspect Scores Over Time")
        fig_ts, ax = plt.subplots(figsize=(12, 4.5))
        for col, color in zip(ASPECT_COLS, ASPECT_COLORS):
            valid = ts[["review_date", col]].dropna()
            if not valid.empty:
                ax.plot(
                    valid["review_date"], valid[col],
                    label=ASPECT_LABELS[col], color=color,
                    linewidth=2, marker="o", markersize=3,
                )
        ax.set_ylabel("Average Rating (1-5)")
        ax.set_ylim(1, 5)
        hotel_sfx = "" if not hotel_id_global.strip() else f" (Hotel {hotel_id_global.strip()})"
        ax.set_title(f"Aspect Ratings -- {granularity}{hotel_sfx}")
        ax.legend(loc="lower left", fontsize=8, ncol=3)
        ax.axhline(3.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        fig_ts.tight_layout()
        st.pyplot(fig_ts)
        plt.close(fig_ts)

        # Review volume over time
        st.markdown("#### Review Volume Over Time")
        vol = (
            fdf2.set_index("review_date")["overall"]
            .resample(freq)
            .count()
            .reset_index()
            .rename(columns={"overall": "n_reviews"})
        )
        fig_vol, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(vol["review_date"], vol["n_reviews"],
                        alpha=0.4, color="#4575b4")
        ax.plot(vol["review_date"], vol["n_reviews"],
                color="#4575b4", linewidth=1.5)
        ax.set_ylabel("Number of Reviews")
        ax.set_title(f"Review Volume -- {granularity}{hotel_sfx}")
        fig_vol.tight_layout()
        st.pyplot(fig_vol)
        plt.close(fig_vol)

        # Per-aspect individual charts
        with st.expander("Individual Aspect Trend Charts"):
            cols_grid = st.columns(3)
            for idx, (col, color) in enumerate(zip(ASPECT_COLS, ASPECT_COLORS)):
                valid = ts[["review_date", col]].dropna()
                fig_ind, ax = plt.subplots(figsize=(5, 3))
                ax.plot(valid["review_date"], valid[col],
                        color=color, linewidth=2)
                ax.fill_between(valid["review_date"], valid[col],
                                alpha=0.15, color=color)
                ax.set_ylim(1, 5)
                ax.set_title(ASPECT_LABELS[col])
                ax.set_ylabel("Avg Rating")
                fig_ind.tight_layout()
                with cols_grid[idx % 3]:
                    st.pyplot(fig_ind)
                plt.close(fig_ind)

        # Summary stats table
        st.markdown("#### Aspect Summary Statistics (Filtered Period)")
        stat_rows = []
        for col in ASPECT_COLS:
            s = fdf2[col].dropna()
            stat_rows.append({
                "Aspect":  ASPECT_LABELS[col],
                "Mean":    round(s.mean(), 3),
                "Median":  round(s.median(), 3),
                "Std Dev": round(s.std(), 3),
                "Min":     round(s.min(), 2),
                "Max":     round(s.max(), 2),
                "Reviews": int(s.count()),
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

# =============================================================================
# Tab 4 -- Textual Feedback Analytics
# =============================================================================
with tab4:
    st.subheader("Textual Feedback Analytics -- Guest Feedback Insights")

    fdf3 = fdf_base

    if fdf3.empty:
        st.warning("No reviews match the selected filters.")
    else:
        try:
            from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

            @st.cache_data
            def extract_title_topics_df(titles: tuple, n: int = 20):
                if not titles:
                    return pd.DataFrame({"topic": [], "count": []})
                cv = CountVectorizer(
                    stop_words="english", max_features=200,
                    ngram_range=(1, 2), min_df=2,
                )
                cv.fit(titles)
                words = cv.get_feature_names_out()
                counts = np.asarray(cv.transform(titles).sum(axis=0)).flatten()
                top_idx = np.argsort(counts)[-n:][::-1]
                return pd.DataFrame({"topic": words[top_idx], "count": counts[top_idx]})

            @st.cache_data
            def extract_pos_neg_topics_df(
                pos_texts: tuple, neg_texts: tuple, n: int = 20
            ):
                all_texts = list(pos_texts) + list(neg_texts)
                if not all_texts:
                    empty = pd.DataFrame({"topic": [], "count": []})
                    return empty, empty
                cv = CountVectorizer(
                    stop_words="english", max_features=400,
                    ngram_range=(1, 2), min_df=2,
                )
                cv.fit(all_texts)
                words = cv.get_feature_names_out()
                pos_counts = (
                    np.asarray(cv.transform(list(pos_texts)).sum(axis=0)).flatten()
                    if pos_texts else np.zeros(len(words))
                )
                neg_counts = (
                    np.asarray(cv.transform(list(neg_texts)).sum(axis=0)).flatten()
                    if neg_texts else np.zeros(len(words))
                )
                pos_top_idx = np.argsort(pos_counts)[-n:][::-1]
                neg_top_idx = np.argsort(neg_counts)[-n:][::-1]
                pos_df = pd.DataFrame({"topic": words[pos_top_idx], "count": pos_counts[pos_top_idx]})
                neg_df = pd.DataFrame({"topic": words[neg_top_idx], "count": neg_counts[neg_top_idx]})
                return pos_df, neg_df

            # Build text inputs from the filtered dataframe
            title_list = tuple(fdf3["title"].dropna().tolist())
            pos_list   = tuple(fdf3.loc[fdf3["overall"] >= 4, "text"].dropna().tolist())
            neg_list   = tuple(fdf3.loc[fdf3["overall"] <= 2, "text"].dropna().tolist())

            scope_label = scope_global

            # Title keywords
            st.markdown("#### Most Frequent Words in Review Titles")
            title_topics = extract_title_topics_df(title_list)
            if title_topics.empty:
                st.info("Not enough title data for the selected filters.")
            else:
                fig_tt, ax = plt.subplots(figsize=(10, 5))
                ax.barh(title_topics["topic"][::-1], title_topics["count"][::-1], color="#74add1")
                ax.set_xlabel("Frequency")
                ax.set_title(f"Top Keywords in Review Titles -- {scope_label}")
                fig_tt.tight_layout()
                st.pyplot(fig_tt)
                plt.close(fig_tt)

            st.markdown("---")

            # Positive / Negative text keywords
            st.markdown("#### Keyword Themes by Sentiment (Review Text)")
            pos_df, neg_df = extract_pos_neg_topics_df(pos_list, neg_list)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top Positive Keywords** (4-5 stars, {len(pos_list):,} reviews)")
                if pos_df.empty:
                    st.info("Not enough positive reviews for the selected filters.")
                else:
                    fig_pos, ax = plt.subplots(figsize=(6, 7))
                    ax.barh(pos_df["topic"][::-1], pos_df["count"][::-1], color="#4575b4")
                    ax.set_xlabel("Frequency")
                    ax.set_title(f"Positive Keywords -- {scope_label}")
                    fig_pos.tight_layout()
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)

            with c2:
                st.markdown(f"**Top Negative Keywords** (1-2 stars, {len(neg_list):,} reviews)")
                if neg_df.empty:
                    st.info("Not enough negative reviews for the selected filters.")
                else:
                    fig_neg, ax = plt.subplots(figsize=(6, 7))
                    ax.barh(neg_df["topic"][::-1], neg_df["count"][::-1], color="#d73027")
                    ax.set_xlabel("Frequency")
                    ax.set_title(f"Negative Keywords -- {scope_label}")
                    fig_neg.tight_layout()
                    st.pyplot(fig_neg)
                    plt.close(fig_neg)

            with st.expander("Interpretation"):
                st.markdown(
                    "- **Positive reviews** gravitate around words like *great, clean, staff, "
                    "comfortable,* and *location*, confirming that service quality and cleanliness "
                    "are the primary drivers of guest delight.\n"
                    "- **Negative reviews** surface words like *dirty, noise, rude,* and "
                    "*disappointing*, highlighting failure modes hotels must address first.\n"
                    "- **Titles** tend to be more concise but mirror these themes, offering a "
                    "fast sentiment signal before reading the full text."
                )

        except ImportError:
            st.warning(
                "Install scikit-learn to enable text analytics: pip install scikit-learn"
            )

# =============================================================================
# Tab 5 -- Top Reviewer Details
# =============================================================================
with tab5:
    st.subheader("Top Reviewer Details")
    st.markdown(
        "Detailed view of the highest-rated reviews, enriched with reviewer "
        "location and contribution statistics."
    )

    fdf4 = fdf_global

    if fdf4.empty:
        st.warning("No reviews match the selected filters.")
    else:
        n_top = st.slider(
            "Number of top reviews to show", 5, 50, 20, key="n_top_t4"
        )

        top_reviews = (
            fdf4.nlargest(n_top, "overall")
            .reset_index(drop=True)
        )

        # ── Summary KPIs ──────────────────────────────────────────────────────
        k1, k2, k3, k4_col = st.columns(4)
        k1.metric("Reviews Shown", f"{len(top_reviews):,}")
        k2.metric("Avg Overall Rating", f"{top_reviews['overall'].mean():.2f}")
        unique_locs = top_reviews["author_location"].dropna().nunique()
        k3.metric("Unique Reviewer Locations", f"{unique_locs:,}")
        avg_rev_count = top_reviews["author_num_reviews"].dropna().mean()
        k4_col.metric("Avg Reviewer Review Count", f"{avg_rev_count:.0f}" if not pd.isna(avg_rev_count) else "N/A")

        st.markdown("---")

        # ── Main detail table ──────────────────────────────────────────────────
        st.markdown("#### Review Details")
        display_cols = [
            "offering_id", "review_date", "overall",
            "title", "author_name", "author_location",
            "author_num_reviews", "author_num_helpful_votes",
        ]
        rename_detail = {
            "offering_id":            "Hotel ID",
            "review_date":            "Date",
            "overall":                "Overall",
            "title":                  "Title",
            "author_name":            "Reviewer Name",
            "author_location":        "Reviewer Location",
            "author_num_reviews":     "Reviewer # Reviews",
            "author_num_helpful_votes": "Helpful Votes",
        }
        detail_df = (
            top_reviews[display_cols]
            .rename(columns=rename_detail)
            .reset_index(drop=True)
        )
        st.dataframe(detail_df, use_container_width=True)

        # ── Reviewer location breakdown ────────────────────────────────────────
        st.markdown("---")
        loc_counts = (
            top_reviews["author_location"]
            .dropna()
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
        )
        if not loc_counts.empty:
            st.markdown("#### Top Reviewer Locations (Top 15)")
            fig_loc, ax = plt.subplots(figsize=(9, max(3, len(loc_counts) * 0.38)))
            ax.barh(loc_counts.index, loc_counts.values, color="#74add1")
            ax.set_xlabel("Number of Reviews")
            ax.set_title("Reviewer Locations in Top Reviews")
            for i, v in enumerate(loc_counts.values):
                ax.text(v + 0.1, i, str(v), va="center", fontsize=8)
            fig_loc.tight_layout()
            st.pyplot(fig_loc)
            plt.close(fig_loc)

        # ── Reviewer experience distribution ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### Reviewer Experience (Total Reviews Written)")
        rev_counts = top_reviews["author_num_reviews"].dropna()
        if not rev_counts.empty:
            bins = [0, 5, 15, 30, 60, 100, float("inf")]
            labels_bins = ["1-5", "6-15", "16-30", "31-60", "61-100", "100+"]
            binned = pd.cut(rev_counts, bins=bins, labels=labels_bins)
            bin_counts = binned.value_counts().reindex(labels_bins).fillna(0).astype(int)
            fig_exp, ax = plt.subplots(figsize=(7, 3))
            ax.bar(bin_counts.index, bin_counts.values, color="#4575b4")
            ax.set_xlabel("Total Reviews Written by Reviewer")
            ax.set_ylabel("Number of Top Reviewers")
            ax.set_title("Experience Level of Reviewers in Top Reviews")
            for i, (label, v) in enumerate(bin_counts.items()):
                if v > 0:
                    ax.text(i, v + max(bin_counts) * 0.01, str(v),
                            ha="center", va="bottom", fontsize=9)
            fig_exp.tight_layout()
            st.pyplot(fig_exp)
            plt.close(fig_exp)

        # ── Full review text expander ──────────────────────────────────────────
        with st.expander("Read Full Review Texts"):
            for _, row in top_reviews.iterrows():
                loc_str = row["author_location"] if pd.notna(row["author_location"]) else "Unknown location"
                name_str = row["author_name"] if pd.notna(row["author_name"]) else "Anonymous"
                num_rev = int(row["author_num_reviews"]) if pd.notna(row["author_num_reviews"]) else "?"
                st.markdown(
                    f"**{row['title']}** — Overall: {row['overall']:.0f}/5 "
                    f"| Hotel {int(row['offering_id'])} "
                    f"| {str(row['review_date'])[:10]}  \n"
                    f"*{name_str}* from **{loc_str}** ({num_rev} reviews total)"
                )
                if pd.notna(row.get("text")):
                    st.write(row["text"])
                st.markdown("---")

# =============================================================================
# Tab 1 -- Overall Rating Distribution
# =============================================================================
with tab1:
    st.subheader("Overall Rating Distribution")

    fdf5 = fdf_base

    if fdf5.empty:
        st.warning("No reviews match the selected filters.")
    else:
        scope_t5 = scope_global

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Reviews", f"{len(fdf5):,}")
        k2.metric("Avg Overall Rating", f"{fdf5['overall'].mean():.2f}")
        pct_high = (fdf5["overall"] >= 4).mean() * 100
        k3.metric("High Ratings (4-5 stars)", f"{pct_high:.1f}%")

        st.markdown("---")

        # Bar chart
        st.markdown("#### Rating Distribution")
        rc = fdf5["overall"].value_counts().sort_index()
        fig_rd, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            rc.index.astype(str), rc.values,
            color=["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"],
        )
        ax.set_xlabel("Overall Rating")
        ax.set_ylabel("Number of Reviews")
        ax.set_title(f"Overall Rating Distribution -- {scope_t5} ({len(fdf5):,} reviews)")
        for bar, v in zip(bars, rc.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rc) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=9,
            )
        fig_rd.tight_layout()
        st.pyplot(fig_rd)
        plt.close(fig_rd)

        # Percentage breakdown table
        st.markdown("#### Percentage Breakdown")
        pct_df = pd.DataFrame({
            "Rating": rc.index.astype(int),
            "Reviews": rc.values,
            "% of Total": (rc.values / rc.values.sum() * 100).round(2),
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # Trend over time
        st.markdown("---")
        st.markdown("#### Rating Trend Over Time")
        granularity_t5 = st.radio(
            "Time granularity",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True,
            key="granularity_t5",
        )
        freq_map_t5 = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        avg_trend = (
            fdf5.set_index("review_date")["overall"]
            .resample(freq_map_t5[granularity_t5])
            .mean()
            .reset_index()
        )
        vol_trend = (
            fdf5.set_index("review_date")["overall"]
            .resample(freq_map_t5[granularity_t5])
            .count()
            .reset_index()
            .rename(columns={"overall": "n_reviews"})
        )
        fig_trend, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(avg_trend["review_date"], avg_trend["overall"],
                 color="#4575b4", linewidth=2, marker="o", markersize=3)
        ax1.axhline(fdf5["overall"].mean(), color="gray", linestyle="--",
                    linewidth=0.9, alpha=0.7, label=f"Mean {fdf5['overall'].mean():.2f}")
        ax1.set_ylabel("Avg Overall Rating")
        ax1.set_ylim(1, 5)
        ax1.set_title(f"Avg Rating & Review Volume -- {granularity_t5} ({scope_t5})")
        ax1.legend(fontsize=8)
        ax2.fill_between(vol_trend["review_date"], vol_trend["n_reviews"],
                         alpha=0.4, color="#74add1")
        ax2.plot(vol_trend["review_date"], vol_trend["n_reviews"],
                 color="#74add1", linewidth=1.5)
        ax2.set_ylabel("Number of Reviews")
        fig_trend.tight_layout()
        st.pyplot(fig_trend)
        plt.close(fig_trend)