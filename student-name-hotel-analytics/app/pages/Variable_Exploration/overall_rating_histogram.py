import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Overall Rating Distribution Histograms & Pie Chart")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

plt.figure(figsize=(10, 6))
sns.histplot(data=df_origin['overall'], bins=5, color='skyblue')
plt.title('Overall Rating Count Distribution')
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.tight_layout()
st.pyplot(plt)

plt.figure(figsize=(8, 8))
overall_counts = df_origin['overall'].value_counts()
plt.pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Overall Rating Distribution Pie Chart')
plt.axis('equal')
st.pyplot(plt)
