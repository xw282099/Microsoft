import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from .data_presentation import load_data


def app():
    st.title("Data Visualization")
    df = load_data()

    st.markdown("### Feature Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="viridis")
    st.pyplot(plt.gcf())
