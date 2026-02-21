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
st.title("Customer Satisfaction Driver Dashboard")
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

# â”€â”€ Data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def load_reviews(_conn):
    df = pd.read_sql(
        """
        SELECT overall, service, cleanliness, value, location_rating,
               sleep_quality, rooms, title, text, review_date
        FROM reviews
        WHERE overall BETWEEN 1 AND 5
          AND review_date IS NOT NULL
        ORDER BY review_date
        """,
        _conn,
    )
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df.dropna(subset=["review_date"])


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

# â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "â­ Rating Breakdown",
        "ðŸ† Top Performers",
        "ðŸ’¬ Top Topics",
        "ðŸ“Š Feature Importance",
    ]
)
with tab1:
    st.subheader("Rating Breakdown")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribution of Overall Ratings (by Reviews)**")
        rating_counts = df["overall"].value_counts().sort_index()
        fig2, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            rating_counts.index.astype(str),
            rating_counts.values,
            color=["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"],
        )
        ax.set_xlabel("Overall Rating")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Rating Distribution")
        for bar, v in zip(bars, rating_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rating_counts) * 0.01,
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_b:
        st.markdown("**Average Score by Rating Type**")
        aspect_means = df[ASPECT_COLS].mean().rename(ASPECT_LABELS).sort_values(ascending=True)
        median_val = aspect_means.median()
        colors_bar = ["#4575b4" if v >= median_val else "#d73027" for v in aspect_means.values]
        fig3, ax = plt.subplots(figsize=(6, 4))
        ax.barh(aspect_means.index, aspect_means.values, color=colors_bar)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Average Rating")
        ax.set_title("Avg Aspect Rating (All Reviews)")
        for i, v in enumerate(aspect_means.values):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")
    st.markdown("**Quarterly Trend by Rating Type**")
    aspect_monthly = (
        df.set_index("review_date")[ASPECT_COLS]
        .resample("QE")
        .mean()
        .reset_index()
    )
    fig4, ax = plt.subplots(figsize=(12, 4))
    for col in ASPECT_COLS:
        valid = aspect_monthly[["review_date", col]].dropna()
        ax.plot(valid["review_date"], valid[col], label=ASPECT_LABELS[col], linewidth=1.8)
    ax.set_ylabel("Avg Rating")
    ax.set_title("Quarterly Aspect Rating Trends")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.set_ylim(1, 5)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tab 3 - Top Performers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab2:
    st.subheader("Top & Bottom Performing Hotels")

    AVG_ASP = ["avg_service", "avg_cleanliness", "avg_value", "avg_location", "avg_sleep_quality", "avg_rooms"]
    DISPLAY  = ["Service", "Cleanliness", "Value", "Location", "Sleep Quality", "Rooms"]
    ASP_RAW  = ["service", "cleanliness", "value", "location_rating", "sleep_quality", "rooms"]

    # ── Filter row ────────────────────────────────────────────────────────────
    filter_col1, filter_col2 = st.columns([1, 1])

    with filter_col1:
        st.markdown("#### Filter 1 — My Hotel")
        st.caption("Enter your hotel's Offering ID to benchmark it against top/bottom performers.")
        my_hotel_id_input = st.text_input("Offering ID", value="", placeholder="e.g. 93466", key="my_hotel_id")
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
                    st.success(f"Hotel {my_hotel_id} loaded — Avg Overall: {my_overall:.2f}")
            except ValueError:
                st.error("Please enter a valid numeric Offering ID.")

    with filter_col2:
        st.markdown("#### Filter 2 — Date Range")
        st.caption("Restrict data to reviews posted within a date range.")
        min_date = df["review_date"].min().date()
        max_date = df["review_date"].max().date()
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_filter",
        )

    st.markdown("---")

    # ── Recompute hotel stats filtered by date range ──────────────────────────
    @st.cache_data
    def hotel_stats_by_daterange(_conn, start_date: str, end_date: str):
        if start_date == str(df["review_date"].min().date()) and end_date == str(df["review_date"].max().date()):
            return hotel_stats
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

    # handle both a complete (start, end) tuple and a single-date selection
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

    hs = hotel_stats_by_daterange(conn, str(start_d), str(end_d))

    is_full_range = (start_d == min_date and end_d == max_date)
    label_sfx = " (all dates)" if is_full_range else f" ({start_d.strftime('%d %b %Y')} – {end_d.strftime('%d %b %Y')})"

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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tab 4 - Top Topics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab3:
    st.subheader("Top Positive & Negative Review Topics")
    st.markdown(
        "Keywords extracted from high-rated (4-5 stars) and low-rated (1-2 stars) reviews "
        "reveal what guests praise and what frustrates them."
    )

    try:
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

        @st.cache_data
        def extract_topics(_conn, n_words: int = 20):
            pos_texts = pd.read_sql(
                "SELECT text FROM reviews WHERE overall >= 4 AND text IS NOT NULL ORDER BY RANDOM() LIMIT 20000",
                _conn,
            )["text"].tolist()
            neg_texts = pd.read_sql(
                "SELECT text FROM reviews WHERE overall <= 2 AND text IS NOT NULL ORDER BY RANDOM() LIMIT 20000",
                _conn,
            )["text"].tolist()

            cv = CountVectorizer(
                stop_words="english", max_features=300, ngram_range=(1, 2), min_df=5
            )
            cv.fit(pos_texts + neg_texts)
            words = cv.get_feature_names_out()
            pos_counts = np.asarray(cv.transform(pos_texts).sum(axis=0)).flatten()
            neg_counts = np.asarray(cv.transform(neg_texts).sum(axis=0)).flatten()

            pos_top = pd.DataFrame(
                {"topic": words[np.argsort(pos_counts)[-n_words:][::-1]],
                 "count": pos_counts[np.argsort(pos_counts)[-n_words:][::-1]]}
            )
            neg_top = pd.DataFrame(
                {"topic": words[np.argsort(neg_counts)[-n_words:][::-1]],
                 "count": neg_counts[np.argsort(neg_counts)[-n_words:][::-1]]}
            )
            return pos_top, neg_top

        pos_df, neg_df = extract_topics(conn)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Top Positive Topics** (Overall 4-5 stars)")
            fig8, ax = plt.subplots(figsize=(6, 7))
            ax.barh(pos_df["topic"][::-1], pos_df["count"][::-1], color="#4575b4")
            ax.set_xlabel("Frequency")
            ax.set_title("Top Positive Keywords")
            fig8.tight_layout()
            st.pyplot(fig8)
            plt.close(fig8)

        with c2:
            st.markdown("**Top Negative Topics** (Overall 1-2 stars)")
            fig9, ax = plt.subplots(figsize=(6, 7))
            ax.barh(neg_df["topic"][::-1], neg_df["count"][::-1], color="#d73027")
            ax.set_xlabel("Frequency")
            ax.set_title("Top Negative Keywords")
            fig9.tight_layout()
            st.pyplot(fig9)
            plt.close(fig9)

        with st.expander("Interpretation"):
            st.markdown(
                "Positive reviews concentrate on words like *great*, *clean*, *staff*, *location*, "
                "and *comfortable*, confirming that service quality and cleanliness are the main "
                "drivers of delight. Negative reviews surface words like *dirty*, *noise*, *rude*, "
                "and *disappointing*, highlighting the failure modes hotels must address first."
            )

    except ImportError:
        st.warning("Install `scikit-learn` to enable topic extraction (`pip install scikit-learn`).")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tab 5 - Feature Importance
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab4:
    st.subheader("Feature Importance: What Drives the Overall Rating?")
    st.info("Cleanliness impacts the overall rating the most - invest here first to see the greatest lift in scores.")

    rated = df[ASPECT_COLS + ["overall"]].dropna()
    corr = (
        rated[ASPECT_COLS]
        .corrwith(rated["overall"])
        .rename(ASPECT_LABELS)
        .sort_values(ascending=True)
    )

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        @st.cache_data
        def get_rf_importance(data: pd.DataFrame):
            sample = data.sample(min(50_000, len(data)), random_state=42)
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(sample[ASPECT_COLS], sample["overall"])
            return rf.feature_importances_

        importances = get_rf_importance(rated)
        imp_series = (
            pd.Series(importances, index=[ASPECT_LABELS[c] for c in ASPECT_COLS])
            .sort_values(ascending=True)
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pearson Correlation with Overall Rating**")
            fig10, ax = plt.subplots(figsize=(6, 4))
            ax.barh(corr.index, corr.values, color="#4575b4")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Correlation coefficient")
            ax.set_title("Aspect â†’ Overall Correlation")
            for i, v in enumerate(corr.values):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
            fig10.tight_layout()
            st.pyplot(fig10)
            plt.close(fig10)

        with c2:
            st.markdown("**Random Forest Feature Importance**")
            fig11, ax = plt.subplots(figsize=(6, 4))
            ax.barh(imp_series.index, imp_series.values, color="#1f77b4")
            ax.set_xlabel("Importance Score")
            ax.set_title("RF Feature Importance (predicting Overall)")
            for i, v in enumerate(imp_series.values):
                ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
            fig11.tight_layout()
            st.pyplot(fig11)
            plt.close(fig11)

        top_driver = imp_series.idxmax()
        st.success(
            f"**Key Insight:** *{top_driver}* is the strongest driver of overall rating. "
            f"Hotels that improve {top_driver.lower()} scores will see the greatest lift "
            "in guest satisfaction."
        )

    except ImportError:
        # Fallback: correlation only
        fig10, ax = plt.subplots(figsize=(8, 4))
        ax.barh(corr.index, corr.values, color="#4575b4")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Correlation coefficient")
        ax.set_title("Aspect â†’ Overall Correlation")
        for i, v in enumerate(corr.values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        fig10.tight_layout()
        st.pyplot(fig10)
        plt.close(fig10)
        st.warning("Install `scikit-learn` for Random Forest importance (`pip install scikit-learn`).")

    st.markdown("---")
    st.markdown("### Business Recommendations")
    top3 = corr.sort_values(ascending=False).head(3)
    labels_map = {1: "highest", 2: "second-highest", 3: "third-highest"}
    for rank, (aspect, val) in enumerate(top3.items(), 1):
        st.markdown(
            f"{rank}. **{aspect}** - correlation **{val:.3f}** with overall rating. "
            f"This aspect delivers the {labels_map[rank]} impact on guest satisfaction."
        )
