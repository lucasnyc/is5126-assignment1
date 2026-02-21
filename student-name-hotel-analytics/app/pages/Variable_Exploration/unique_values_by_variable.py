import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Variable Unique Values")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

uniq_count = df_origin.nunique()

# Align variable types to the columns present in df_origin
var_types = (
    df_variables.set_index("name")["type"]
    .reindex(uniq_count.index)
)

df_uniq_count = pd.DataFrame({
    "Unique Values": uniq_count,
    "Type": var_types
})

st.dataframe(df_uniq_count, use_container_width=True, height=600)