import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def load_model():
    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def app():
    st.title("Feature Importance")
    model, X_test = load_model()

    st.markdown("### SHAP Summary")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(plt.gcf(), bbox_inches="tight", dpi=300)
