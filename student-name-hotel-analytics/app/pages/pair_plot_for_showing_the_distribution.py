import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Pair Plot for Showing the Distribution")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

df_origin_sample = df_origin.sample(frac=0.2)
g=sns.pairplot(df_origin_sample)
plt.figure(figsize=(20, 60))
plt.suptitle('Pair Plot for DataFrame')
st.pyplot(g.figure)