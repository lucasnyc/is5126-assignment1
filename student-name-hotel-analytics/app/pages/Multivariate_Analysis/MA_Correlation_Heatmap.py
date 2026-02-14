import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Correlation Heatmap")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]
df_variables_integer = df_variables[df_variables["type"] == "Int64"]["name"]

plt.figure(figsize=(15, 10))
sns.heatmap(df_origin[df_variables_integer].corr(), annot=True, fmt='.2f', cmap='Blues', linewidths=2)
plt.title('Correlation Heatmap')
st.pyplot(plt)