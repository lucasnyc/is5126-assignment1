import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Scatter Plots of features vs Overall Review Status")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]
overall_review_statuses = df_origin['overall'].unique()

tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(['service','cleanliness','value', 'location_rating','sleep_quality','rooms'])
with tab1:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='service']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='service', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='service', y='overall', scatter=False, color='red')
    plt.xlabel('Service')
    plt.ylabel("Overall Review Status")
    plt.title(f'Service vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab2:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='cleanliness']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='cleanliness', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='cleanliness', y='overall', scatter=False, color='red')
    plt.xlabel('Cleanliness')
    plt.ylabel("Overall Review Status")
    plt.title(f'Cleanliness vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab3:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='value']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='value', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='value', y='overall', scatter=False, color='red')
    plt.xlabel('Value')
    plt.ylabel("Overall Review Status")
    plt.title(f'Value vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab4:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='location_rating']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='location_rating', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='location_rating', y='overall', scatter=False, color='red')
    plt.xlabel('Location Rating')
    plt.ylabel("Overall Review Status")
    plt.title(f'Location Rating vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab5:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='sleep_quality']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='sleep_quality', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='sleep_quality', y='overall', scatter=False, color='red')
    plt.xlabel('Sleep Quality')
    plt.ylabel("Overall Review Status")
    plt.title(f'Sleep Quality vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab6:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='rooms']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='rooms', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='rooms', y='overall', scatter=False, color='red')
    plt.xlabel('Rooms')
    plt.ylabel("Overall Review Status")
    plt.title(f'Rooms vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)