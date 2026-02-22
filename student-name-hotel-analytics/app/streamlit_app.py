import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3

# Get project root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT 
st.set_page_config(layout="wide")

DATA_DIR = PROJECT_ROOT / "data"
JSON_FILE = DATA_DIR / "review.json"        
DB_PATH = DATA_DIR / "reviews_sample.db"
SQL_SCHEMA = DATA_DIR / "data_schema.sql"
SQL_INDEXING = DATA_DIR / "data_indexing.sql"

df_variables = pd.read_csv(DATA_DIR / "consolidated_data_variables_info.csv")

_conn = sqlite3.connect(str(DB_PATH))
df_origin = pd.read_sql("""
    SELECT
        r.*,
        a.author_name,
        a.author_location,
        a.author_num_reviews,
        a.author_num_cities,
        a.author_num_helpful_votes AS author_helpful_votes,
        a.author_num_type_reviews
    FROM reviews r
    LEFT JOIN authors a ON r.author_id = a.author_id
    LEFT JOIN hotels h ON r.offering_id = h.offering_id
""", _conn)
_conn.close()

st.session_state["DB_PATH"] = str(DB_PATH)
st.session_state["df_variables"] = df_variables
st.session_state["df_origin"] = df_origin
show_pages = {
    "Variable Explorations":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "variable_exploration.py",title="Variable Overview",icon="🗒️"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "missing_values_by_variable.py",title="Missing Values by Variable",icon="❓"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "variable_frequency_table.py",title="Variable Frequency Table",icon="📋"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "descriptive_stats_table.py",title="Descriptive Statistics Table",icon="📈"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "unique_values_by_variable.py",title="Unique Values by Variable",icon="🔢"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Variable_Exploration" / "overall_rating_histogram.py",title="Target Variable Histogram",icon="📊"),
    ],
    "Univariate Analysis by Category":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Univariate_Analysis" / "UA_Author_Info.py",title="Author Info KDE Plot",icon="📊"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Univariate_Analysis" / "UA_Review_Info.py",title="Review Info Histogram",icon="📊"),
    ],
    "Bivariate Analysis":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Author_Behavior.py",title="Author Behavior Bivariate Analysis",icon="📉"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Hotel_Info.py",title="Hotel Info Based Bivariate Analysis",icon="📉"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Review_Info.py",title="Review Info Based Bivariate Analysis",icon="📉"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Time_Series_Analysis.py",title="Time Series Analysis",icon="📉"),
    ],
    "Multivariate Analysis":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Multivariate_Analysis" /"MA_Correlation_Heatmap.py",title="Correlation Heatmap",icon="🔢"),
    ],
    "Intelligent Analytics Platform":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Intelligent_Analytics_Platform" / "Guest_Experience_Dashboard.py", title="My Hotel Guest Experience Dashboard", icon="🧠"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Intelligent_Analytics_Platform" / "Market_Position_Dashboard.py", title="My Hotel Market Position Dashboard", icon="🧠"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Intelligent_Analytics_Platform" / "Improvement_Plan_by_LLM.py", title="My Hotel Improvement Plan Dashboard", icon="🧠"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Intelligent_Analytics_Platform" / "Investment_Return_Prediction.py", title="My Hotel Investment Return Prediction Dashboard", icon="🧠"),

    ],

}

pg =st.navigation(show_pages)
pg.run()