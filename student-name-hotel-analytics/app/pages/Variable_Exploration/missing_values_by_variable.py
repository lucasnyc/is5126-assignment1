import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Missing Values by Variable")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

# st.markdown(""" **Findings and Observations**:  
# - weight data suffers from extensive missing values, with nearly 100,000 entries of ‘Nan’. Among recorded patients, the analysis of the value shows a reverse trend between weight and early readmission. """)
st.markdown("---")
rows = []
for col in df_origin.columns:
    rows.append({"Column": col, "Missing Values Qty": df_origin[col].isnull().sum(), "Missing Values %": round(df_origin[col].isnull().sum()/len(df_origin)*100,2)})
df_missing_values=pd.DataFrame(rows)
st.dataframe(df_missing_values, use_container_width=True, height=600)