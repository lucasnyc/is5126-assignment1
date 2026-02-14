import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Scatter Plots of features vs Overall Review Status")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]
overall_review_statuses = df_origin['overall'].unique()

tab1, tab2, tab3, tab4 = st.tabs([ 'author_num_reviews', 'author_num_cities','author_num_helpful_votes','author_num_type_reviews'])
with tab1:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_reviews']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='author_num_reviews', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='author_num_reviews', y='overall', scatter=False, color='red')
    plt.xlabel('Author Number of Reviews')
    plt.ylabel("Overall Review Status")
    plt.title(f'Author Number of Reviews vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab2:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_cities']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='author_num_cities', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='author_num_cities', y='overall', scatter=False, color='red')
    plt.xlabel('Author Number of Cities')
    plt.ylabel("Overall Review Status")
    plt.title(f'Author Number of Cities vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab3:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_helpful_votes']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='author_num_helpful_votes', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='author_num_helpful_votes', y='overall', scatter=False, color='red')
    plt.xlabel('Author Number of Helpful Votes')
    plt.ylabel("Overall Review Status")
    plt.title(f'Author Number of Helpful Votes vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
with tab4:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_type_reviews']['description']}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_origin, x='author_num_type_reviews', y='overall', hue='overall', palette='Set2')
    sns.regplot(data=df_origin, x='author_num_type_reviews', y='overall', scatter=False, color='red')
    plt.xlabel('Author Number of Type Reviews')
    plt.ylabel("Overall Review Status")
    plt.title(f'Author Number of Type Reviews vs Overall Review Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)