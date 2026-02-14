import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Author Info Distribution Histograms")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

# tab1, tab2, tab3, tab4, tab5 = st.tabs([ 'author_num_reviews', 'author_num_cities','author_num_helpful_votes','author_num_type_reviews','author_location'])
tab1, tab2, tab3, tab4 = st.tabs([ 'author_num_reviews', 'author_num_cities','author_num_helpful_votes','author_num_type_reviews'])

with tab1:
    series = df_origin['author_num_reviews']
    skewness = series.skew()
    mean = series.mean()
    median = series.median()
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_reviews']['description']}")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_origin['author_num_reviews'], fill=True, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.1f}")
    plt.axvline(median, color='green', linestyle='--', label=f"Median = {median:.1f}")
    plt.title(f'Author Number of Reviews Count Distribution (skewness = {skewness:.2f})')
    plt.xlabel('Author Number of Reviews')
    plt.ylabel('Density')
    plt.tight_layout()
    st.pyplot(plt)

with tab2:
    series = df_origin['author_num_cities']
    skewness = series.skew()
    mean = series.mean()
    median = series.median()
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_cities']['description']}")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_origin['author_num_cities'], fill=True, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.1f}")
    plt.axvline(median, color='green', linestyle='--', label=f"Median = {median:.1f}")
    plt.title(f'Author Number of Cities Count Distribution (skewness = {skewness:.2f})')
    plt.xlabel('Author Number of Cities')
    plt.ylabel('Density')
    plt.tight_layout()
    st.pyplot(plt)

with tab3:
    series = df_origin['author_num_helpful_votes']
    skewness = series.skew()
    mean = series.mean()
    median = series.median()
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_helpful_votes']['description']}")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_origin['author_num_helpful_votes'], fill=True, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.1f}")
    plt.axvline(median, color='green', linestyle='--', label=f"Median = {median:.1f}")
    plt.xlabel('Author Number of Helpful Votes')
    plt.ylabel('Density')
    plt.title(f'Author Number of Helpful Votes Count Distribution (skewness = {skewness:.2f})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab4:
    series = df_origin['author_num_type_reviews']
    skewness = series.skew()
    mean = series.mean()
    median = series.median()
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='author_num_type_reviews']['description']}")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_origin['author_num_type_reviews'], fill=True, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.1f}")
    plt.axvline(median, color='green', linestyle='--', label=f"Median = {median:.1f}")
    plt.xlabel('Author Number of Type Reviews')
    plt.ylabel('Density')
    plt.title(f'Author Number of Type Reviews Count Distribution (skewness = {skewness:.2f})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
