import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Descriptive Statistics Table")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

df_desc_stats=df_origin[[i for i in df_variables[(df_variables["type"]=="Int64") | (df_variables["type"]=="float32")]["name"]]].describe().round(2).T
st.dataframe(df_desc_stats, use_container_width=True, height=600)