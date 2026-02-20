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
st.title("Author Behavior Bivariate Analysis")

author_stats = pd.read_sql("""
SELECT
  author_id,
  COUNT(*) AS n_reviews,
  AVG(overall) AS avg_overall
FROM reviews
WHERE author_id IS NOT NULL AND author_id <> '' AND overall BETWEEN 1 AND 5
GROUP BY author_id
HAVING COUNT(*) >= 2;
""", conn)
author_stats.head()

tab1, tab2 = st.tabs([ 'Distribution of reviews per author', 'Author activity vs avg rating tendency'])
with tab1:
    st.markdown("The distribution of reviews per author is highly right-skewed, with the vast majority of authors contributing only a small number of reviews and a rapidly decreasing number of highly active reviewers. This long-tail pattern indicates that overall review content is dominated by occasional contributors, while a small subset of authors accounts for a disproportionate share of reviews.")
    plt.figure(figsize=(10, 6))
    plt.hist(author_stats["n_reviews"], bins=50)
    plt.yscale("log")
    plt.title("Distribution of reviews per author (authors with >=2 reviews)")
    plt.tight_layout()
    st.pyplot(plt)
with tab2:
    st.markdown("The relationship between author activity and average rating shows **no strong systematic bias**, with most authors—regardless of activity level—clustered around average ratings of approximately 3.5–4.5. While less active authors exhibit greater variability in rating tendencies (likely due to small sample sizes), highly active authors appear more stable, suggesting their average ratings converge as contribution volume increases.")
    plt.figure(figsize=(10, 6))
    plt.scatter(author_stats["n_reviews"], author_stats["avg_overall"], s=5)
    plt.xscale("log")
    plt.title("Author activity vs avg rating tendency")
    plt.xlabel("n_reviews (log scale)")
    plt.ylabel("avg_overall")
    plt.tight_layout()
    st.pyplot(plt)