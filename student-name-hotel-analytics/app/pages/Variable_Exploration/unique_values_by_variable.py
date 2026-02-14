import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Variable Unique Values")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

uniq_count=df_origin.nunique()
df_uniq_count=pd.DataFrame({
    "Unique Values": uniq_count,
    "Type": df_variables["type"].values
})

st.dataframe(df_uniq_count, use_container_width=True, height=600)