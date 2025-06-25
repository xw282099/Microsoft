import streamlit as st
from pages import data_presentation, data_visualization, model_prediction, feature_importance, hyperparameter_tuning

# Streamlit multipage app
st.set_page_config(page_title="Business Case App", layout="wide")

PAGES = {
    "Business Case & Data": data_presentation,
    "Data Visualization": data_visualization,
    "Model Prediction": model_prediction,
    "Feature Importance": feature_importance,
    "Hyperparameter Tuning": hyperparameter_tuning,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
