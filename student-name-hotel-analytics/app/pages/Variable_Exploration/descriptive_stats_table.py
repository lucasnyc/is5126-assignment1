import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Descriptive Statistics Table")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

# Select numeric variables based on df_variables, but only keep those present in df_origin
numeric_var_names = df_variables[
	(df_variables["type"] == "Int64") | (df_variables["type"] == "float32")
]["name"].tolist()

numeric_cols = [name for name in numeric_var_names if name in df_origin.columns]

df_desc_stats = df_origin[numeric_cols].describe().round(2).T if numeric_cols else pd.DataFrame()
st.dataframe(df_desc_stats, use_container_width=True, height=600)