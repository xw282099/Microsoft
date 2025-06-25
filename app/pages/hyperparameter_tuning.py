import streamlit as st
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def train_with_mlflow(n_estimators):
    with mlflow.start_run():
        dataset = fetch_california_housing(as_frame=True)
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mse", mse)
        return mse


def app():
    st.title("Hyperparameter Tuning")
    n_estimators = st.slider("n_estimators", 10, 200, 100, 10)
    if st.button("Run Experiment"):
        mse = train_with_mlflow(n_estimators)
        st.write(f"Logged MSE: {mse:.2f}")
