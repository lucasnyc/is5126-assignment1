from ast import While
import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path

if "DB_PATH" in st.session_state:
    DB_PATH = st.session_state["DB_PATH"]
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DB_PATH = PROJECT_ROOT / "data" / "reviews_sqlite.db"

conn = sqlite3.connect(DB_PATH)
st.title("Hotel Info Based Bivariate Analysis")

hotel_stats = pd.read_sql("""
SELECT
  offering_id,
  COUNT(*) AS n_reviews,
  AVG(overall) AS avg_overall,
  AVG(service) AS avg_service,
  AVG(cleanliness) AS avg_cleanliness,
  AVG(value) AS avg_value,
  AVG(location_rating) AS avg_location,
  (AVG(overall*overall) - AVG(overall)*AVG(overall)) AS var_overall
FROM reviews
WHERE overall BETWEEN 1 AND 5
GROUP BY offering_id;
""", conn)
# >= 50 reviews is a stability threshold. It filters out hotels whose average rating is statistically unreliable because they have too few reviews.
stable_hotels = hotel_stats[hotel_stats["n_reviews"] >= 50].copy()
stable_hotels.head()


tab1, tab2, tab3, tab4,tab5 = st.tabs([ 'Distribution of hotel avg overall rating', 'Distribution of reviews per hotel',
                                  'Hotel volume vs avg overall rating','Top 10 and bottom 10 hotels by avg overall rating', 'Average aspect ratings across hotels'])
with tab1:
    st.markdown("Across hotels with at least 50 reviews, average overall ratings are concentrated in the ~3.5–4.5 range, with a clear peak around the low-4s. There is a small left tail of poorly performing hotels (down to ~1.7–2.5), indicating a minority of properties with consistently weak guest experience.")
    plt.figure(figsize=(10, 6))
    plt.hist(stable_hotels["avg_overall"], bins=30)
    plt.xlabel('Average Overall Rating')
    plt.ylabel("Hotel Count")
    plt.title("Distribution of hotel avg overall rating (n_reviews >= 50)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab2:
    st.markdown("Review counts per hotel are highly right-skewed (long-tail): most hotels have relatively few reviews, while a small subset accumulates very large volumes (into the thousands). This supports using a minimum review threshold for fair benchmarking and also suggests that high-volume hotels will dominate aggregate trends if not segmented.")
    plt.figure(figsize=(10, 6))
    plt.hist(hotel_stats["n_reviews"], bins=50)
    plt.yscale("log")
    plt.title("Distribution of reviews per hotel (log scale)")
    plt.tight_layout()
    st.pyplot(plt)
with tab3:
    st.markdown("The scatter plot shows no strong positive relationship between review volume and average overall rating—high-volume hotels span a wide band of ratings rather than clustering at the top. The densest region is at lower review counts, where ratings are more dispersed (consistent with noisier estimates), while higher-volume hotels appear slightly more stable but still vary meaningfully in quality.")
    plt.figure(figsize=(10, 6))
    plt.scatter(stable_hotels["n_reviews"], stable_hotels["avg_overall"], s=10)
    plt.xlabel("n_reviews")
    plt.ylabel("avg_overall")
    plt.title("Hotel volume vs avg overall rating (n_reviews >= 50)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab4:
    top10 = stable_hotels.sort_values("avg_overall", ascending=False).head(10)
    bottom10 = stable_hotels.sort_values("avg_overall").head(10)

    st.markdown("Top and bottom 10 hotels by average overall rating (among hotels with at least 50 reviews). This helps identify consistently high- and low-performing properties.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 hotels")
        st.dataframe(top10)

    with col2:
        st.subheader("Bottom 10 hotels")
        st.dataframe(bottom10)

    with st.expander("Comments on top and bottom performers"):
        st.markdown(
            """**Top performers**

The highest-rated hotels achieve avg_overall ≈ 4.79–4.87 with review volumes ranging roughly from ~80 to ~1,400+ reviews, indicating these are not “small-sample winners” only. Across these top hotels, cleanliness and service are consistently very high, while value is comparatively lower than cleanliness/service even among top performers—suggesting that “excellent stays” do not always translate into “excellent value-for-money.”

**Bottom performers**

The lowest-rated hotels have avg_overall ≈ 1.68–1.86 despite meeting the minimum review threshold (roughly ~50–144 reviews), implying persistent issues rather than isolated bad experiences. Their aspect ratings are uniformly low (service/cleanliness/value mostly near ~2), and several show very high variance, indicating inconsistent experiences and/or polarization among reviewers.
"""
        )
with tab5:
    aspect_cols = ["avg_service","avg_cleanliness","avg_value","avg_location"]
    aspect_means = stable_hotels[aspect_cols].mean().sort_values()
    st.markdown("The top 10 hotels by average overall rating (among those with at least 50 reviews) achieve ratings in the ~4.79–4.87 range, demonstrating that consistently excellent performance is attainable even with substantial review volumes (ranging from ~80 to ~1,400+). Cleanliness and service are the strongest aspects among top performers, while value tends to lag behind, suggesting that exceptional stays do not always equate to exceptional value-for-money.")
    plt.figure(figsize=(10, 6))
    plt.bar(aspect_means.index, aspect_means.values)
    plt.title("Average aspect ratings across hotels (n_reviews >= 50)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)