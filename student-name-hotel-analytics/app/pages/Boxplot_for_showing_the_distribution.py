import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Box Plot for Showing the Distribution")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]
df_variables_integer = df_variables[df_variables["type"] == "Integer"]["name"]

df_origin_sample = df_origin.sample(frac=0.2)
ncols = 2
nrows = int(np.ceil(len(df_variables_integer)//ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6*nrows))
for idx,col in enumerate(df_variables_integer):
    r = idx // ncols
    c = idx % ncols
    sns.boxplot(x="readmitted", y=col, data=df_origin_sample, ax=axes[r,c], hue="readmitted")
    axes[r,c].set_title(f'Box Plot for Readmission and {col}')
    axes[r,c].set_xlabel('Readmission')
    axes[r,c].set_ylabel(col)
plt.tight_layout()
st.pyplot(fig)