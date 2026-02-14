import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get project root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT 
st.set_page_config(layout="wide")
df_variables = pd.read_csv(PROJECT_ROOT / "data" / "consolidated_data_variables_info.csv")
df_origin = pd.read_csv(PROJECT_ROOT / "data" / "consolidated_data.csv")

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
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Author_Info.py",title="Author Info Bivariate Analysis",icon="📉"),
        st.Page(PROJECT_ROOT / "app" / "pages" / "Bivariate_Analysis" / "BA_Review_Info.py",title="Review Info Bivariate Analysis",icon="📉"),
    ],
    "Multivariate Analysis":[
        st.Page(PROJECT_ROOT / "app" / "pages" / "Multivariate_Analysis" /"MA_Correlation_Heatmap.py",title="Correlation Heatmap",icon="🔢"),
    ],
    # "Other Key Insights":[
    #     st.Page(ROOT / "streamlit" / "pages" / "Key_Insights" /"readmission_vs_medical_specialty.py",title="Medical Specialty VS Readmission",icon="🧠"),
    #     st.Page(ROOT / "streamlit" / "pages" / "Key_Insights" /"readmission_vs_insulin.py",title="Insulin VS Readmission",icon="🧠"),
    # ],

}

pg =st.navigation(show_pages)
pg.run()