import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    return df

def app():
    st.title("Business Case & Data")
    st.markdown(
        """
        ## Business Problem
        This demo app predicts California housing prices based on various
        socio-economic indicators. The goal is to illustrate how machine
        learning can help decision makers understand the housing market.
        """
    )
    st.markdown("### Dataset Preview")
    df = load_data()
    st.write(df.head())
