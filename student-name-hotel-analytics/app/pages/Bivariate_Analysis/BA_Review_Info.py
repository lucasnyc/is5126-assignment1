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
    DB_PATH = PROJECT_ROOT / "data" / "reviews_sample.db"

conn = sqlite3.connect(DB_PATH)
st.title("Review Info Based Bivariate Analysis")

review_sample = pd.read_sql("""
SELECT overall, service, cleanliness, value, location_rating, sleep_quality, rooms,
       title, text, review_date, via_mobile
FROM reviews
WHERE text IS NOT NULL AND overall BETWEEN 1 AND 5 AND review_date IS NOT NULL
ORDER BY RANDOM()
LIMIT 50000;
""", conn)

review_sample["len_words"] = review_sample["text"].str.split().str.len()
review_sample["review_date"] = pd.to_datetime(review_sample["review_date"])

tab1, tab2 = st.tabs([ 'Review length distribution', 'Review length vs overall rating'])
with tab1:
    st.markdown("Review lengths exhibit a highly right-skewed distribution, with most reviews being relatively short and a rapidly diminishing number of very long reviews extending beyond 1,000 words. The long tail suggests that while the majority of users provide brief feedback, a small subset of reviewers contribute detailed, narrative-style reviews, which may contain richer qualitative insights but should be treated carefully to avoid over-weighting verbose outliers in text analysis.")
    plt.figure(figsize=(10, 6))
    plt.hist(review_sample["len_words"], bins=60)
    plt.yscale("log")
    plt.title("Review length distribution (words, log y)")
    plt.tight_layout()
    st.pyplot(plt)
with tab2:
    st.markdown("The relationship between review length and overall rating shows no strong linear correlation, with both short and long reviews appearing across all rating levels. However, lower ratings tend to exhibit greater variability in review length, including a higher concentration of very long reviews, suggesting that dissatisfied guests are more likely to provide detailed explanations. In contrast, higher ratings are more frequently associated with shorter reviews, indicating that positive experiences are often communicated more succinctly.")
    plt.figure(figsize=(10, 6))
    plt.scatter(review_sample["overall"], review_sample["len_words"], s=5)
    plt.title("Review length vs overall rating (50k sample)")
    plt.xlabel("overall")
    plt.ylabel("len_words")
    plt.tight_layout()
    st.pyplot(plt)
