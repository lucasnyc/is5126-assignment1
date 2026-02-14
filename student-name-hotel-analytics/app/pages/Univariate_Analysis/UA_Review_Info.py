import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Review Rating Distribution Histograms")

df_variables = st.session_state["df_variables"]
df_origin = st.session_state["df_origin"]

# tab1, tab2, tab3, tab4, tab5 = st.tabs([ 'author_num_reviews', 'author_num_cities','author_num_helpful_votes','author_num_type_reviews','author_location'])
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([ 'overall', 'service','cleanliness','value', 'location_rating','sleep_quality','rooms'])

with tab1:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='overall']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['overall'], kde=True, color='skyblue')
    plt.title(f'Overall Rating Distribution')
    plt.xlabel('Overall Rating')
    plt.ylabel('Density')
    plt.tight_layout()
    st.pyplot(plt)

with tab2:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='service']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['service'], kde=True, color='skyblue')
    plt.title(f'Service Rating Distribution')
    plt.xlabel('Service Rating')
    plt.ylabel('Density')
    plt.tight_layout()
    st.pyplot(plt)

with tab3:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='cleanliness']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['cleanliness'], kde=True, color='skyblue')
    plt.title(f'Cleanliness Rating Distribution')
    plt.xlabel('Cleanliness Rating')
    plt.ylabel('Density')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab4:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='value']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['value'], kde=True, color='skyblue')
    plt.xlabel('Value Rating')
    plt.ylabel('Density')
    plt.title(f'Value Rating Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab5:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='location_rating']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['location_rating'], kde=True, color='skyblue')
    plt.xlabel('Location Rating')
    plt.ylabel('Density')
    plt.title(f'Location Rating Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab6:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='sleep_quality']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['sleep_quality'], kde=True, color='skyblue')
    plt.xlabel('Sleep Quality Rating')
    plt.ylabel('Density')
    plt.title(f'Sleep Quality Rating Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

with tab7:
    st.markdown(f"**Definition:** {df_variables[df_variables['name']=='rooms']['description']}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_origin['rooms'], kde=True, color='skyblue')
    plt.xlabel('Rooms Rating')
    plt.ylabel('Density')
    plt.title(f'Rooms Rating Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
