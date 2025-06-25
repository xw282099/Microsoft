import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


@st.cache_data
def load_data():
    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)

    return {
        "Linear Regression": (lr, lr_mse),
        "Random Forest": (rf, rf_mse),
    }


def app():
    st.title("Model Prediction")
    X_train, X_test, y_train, y_test = load_data()
    models = train_models(X_train, X_test, y_train, y_test)

    model_name = st.selectbox("Select Model", list(models.keys()))
    model, mse = models[model_name]
    st.write(f"Model Mean Squared Error: {mse:.2f}")

    if st.checkbox("Show predictions on test set"):
        preds = model.predict(X_test)
        df_pred = pd.DataFrame({"y_true": y_test, "y_pred": preds})
        st.write(df_pred.head())
