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
st.title("Time Series Analysis")

monthly = pd.read_sql("""
SELECT
  strftime('%Y-%m', review_date) AS ym,
  COUNT(*) AS review_cnt,
  AVG(overall) AS avg_overall,
  AVG(service) AS avg_service,
  AVG(cleanliness) AS avg_cleanliness,
  AVG(value) AS avg_value,
  AVG(location_rating) AS avg_location
FROM reviews
WHERE review_date IS NOT NULL AND overall IS NOT NULL
GROUP BY ym
ORDER BY ym;
""", conn)

monthly["ym"] = pd.to_datetime(monthly["ym"] + "-01")
monthly.head()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ 'Review volume over time (monthly)', 'Average overall rating over time (monthly)',
                                  'Average service over time (monthly)','Average cleanliness over time (monthly)',
                                  'Average value over time (monthly)','Average location rating over time (monthly)'])
with tab1:
    st.markdown("Monthly review volume increases sharply from 2008 to 2012, indicating rapid growth in platform usage and/or hotel coverage. While there are mild month-to-month fluctuations that suggest seasonality, the dominant pattern is a strong upward trend. The sharp drop at the end of the series is likely due to an incomplete final month rather than a true decline.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["review_cnt"])
    plt.xlabel('Month/Year')
    plt.ylabel("Review Count")
    plt.title("Review volume over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab2:
    st.markdown("Average overall ratings rise from approximately 3.75 in 2008 to around 4.0 and above after 2010, after which they remain relatively stable. This upward shift suggests improving guest satisfaction, changes in hotel mix, or evolving reviewer behavior. Short-term fluctuations persist, but no sustained decline is observed.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["avg_overall"])
    plt.xlabel('Month/Year')
    plt.ylabel("Average Overall Rating")
    plt.title("Average overall rating over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab3:
    st.markdown("Service and cleanliness show steady upward trends over time, suggesting gradual improvements in operational quality that contribute positively to overall guest satisfaction.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["avg_service"])
    plt.xlabel('Month/Year')
    plt.ylabel("Average Service Rating")
    plt.title("Average service rating over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab4:
    st.markdown("Service and cleanliness show steady upward trends over time, suggesting gradual improvements in operational quality that contribute positively to overall guest satisfaction.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["avg_cleanliness"])
    plt.xlabel('Month/Year')
    plt.ylabel("Average Cleanliness Rating")
    plt.title("Average cleanliness rating over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab5:
    st.markdown("Value exhibits the greatest volatility, with noticeable cyclical fluctuations, implying higher sensitivity to pricing, seasonality, and changing customer expectations.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["avg_value"])
    plt.xlabel('Month/Year')
    plt.ylabel("Average Value Rating")
    plt.title("Average value rating over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab6:
    st.markdown("Location is consistently the highest-rated aspect with low volatility, indicating it is a stable strength across hotels and largely unaffected by temporal factors.")
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ym"], monthly["avg_location"])
    plt.xlabel('Month/Year')
    plt.ylabel("Average Location Rating")
    plt.title("Average location rating over time (monthly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)