import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import shap
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

st.set_page_config(page_title="Microsoft Stock Dashboard 📈", layout="wide", page_icon="📊")
st.sidebar.title("Microsoft Stock Explorer 💻")

@st.cache_data
def load_data():
    return pd.read_csv("msft.csv", parse_dates=["Date"])

df = load_data()

def main():
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "1️⃣ Introduction & Data Presentation",
            "2️⃣ Data Visualization",
            "3️⃣ Prediction",
            "4️⃣ Explainability",
            "5️⃣ Hyperparameter Tuning"
        ]
    )

    if page == "1️⃣ Introduction & Data Presentation":
        st.header("1️⃣ Business Case & Data Presentation")
        st.markdown(
            f"""
**Business Case:**  
Predict Microsoft closing stock price to help investors time their trades and manage risk.  
**Impact:**  
Enhances data-driven decision-making, potentially improving portfolio performance.  
**Dataset:**  
- Source: Historical MSFT daily prices  
- Records: {len(df)} days  
- Features: {', '.join(df.columns.drop('Date'))}
"""
        )
        with st.expander("Show Data Preview"):
            n = st.slider("Rows to display", min_value=10, max_value=len(df), value=50)
            st.dataframe(df.head(n))

        st.image('Microsoft.jpg', caption='Microsoft Headquarters')
        video_url = "https://www.youtube.com/watch?v=qKG8r1NERl4"
        st.video(video_url)

        st.markdown("**Summary Statistics**")
        st.dataframe(df.describe())

    elif page == "2️⃣ Data Visualization":
        st.header("2️⃣ Exploratory Data Visualization")
        cols = df.columns.drop("Date")
        col_x = st.selectbox("X-axis", ["Date"] + list(cols), index=0)
        col_y = st.selectbox("Y-axis", list(cols), index=list(cols).index("Close"))

        tab_line, tab_bar, tab_heat, tab_profile = st.tabs([
            "Line Chart 📈", "Bar Chart 📊", "Correlation Heatmap 🔥", "Automated Profile 📑"
        ])

        with tab_line:
            st.subheader("Line Chart")
            st.line_chart(df.set_index(col_x)[col_y])

        with tab_bar:
            st.subheader("Bar Chart")
            st.bar_chart(df.set_index(col_x)[col_y])

        with tab_heat:
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with tab_profile:
            st.subheader("Automated Data Profile")
            if st.button("Generate Profile Report"):
                profile = ProfileReport(df, title="MSFT Profile Report", explorative=True, minimal=True)
                st_profile_report(profile)
                html = profile.to_html()
                st.download_button("📥 Download Full HTML Report", html, "msft_profile.html", "text/html")

    elif page == "3️⃣ Prediction":
        st.header("3️⃣ Prediction: Compare Models")
        df2 = df.dropna().copy()
        feature_opts = list(df2.columns.drop("Date"))
        features = st.multiselect("Select Features (X)", feature_opts, default=["Open", "High", "Low", "Volume"])
        target = st.selectbox("Select Target (y)", feature_opts, index=feature_opts.index("Close"))

        
        ridge_alpha = st.slider("Ridge α", min_value=0.0, max_value=10.0, value=1.0)
        rf_estimators = st.slider("RF n_estimators", min_value=50, max_value=200, value=100)
        model_choice = st.selectbox("Choose Model", ["LinearRegression", "Ridge", "RandomForest"])
        if model_choice == "LinearRegression":
            model = LinearRegression()
        elif model_choice == "Ridge":
            model = Ridge(alpha=ridge_alpha)
        else:
            model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)

        X = df2[features]
        y = df2[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = metrics.r2_score(y_test, preds)
        mae = metrics.mean_absolute_error(y_test, preds)
        mse = metrics.mean_squared_error(y_test, preds)
        st.subheader(f"Model: {model_choice}")
        st.write(f"- **R²:** {r2:.3f}")
        st.write(f"- **MAE:** {mae:.2f}")
        st.write(f"- **MSE:** {mse:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    elif page == "4️⃣ Explainability":
        st.header("4️⃣ Explainable AI with SHAP")
        df2 = df.dropna().copy()
        features_exp = st.multiselect("Features for Explanation", list(df2.columns.drop("Date")), default=["Open", "High", "Low", "Volume"])
        target_exp = st.selectbox("Select Target for Explanation", list(df2.columns.drop("Date")), index=list(df2.columns.drop("Date")).index("Close"))

        X_exp = df2[features_exp]
        y_exp = df2[target_exp]
        expl_model = LinearRegression().fit(X_exp, y_exp)

        explainer = shap.Explainer(expl_model, X_exp)
        shap_values = explainer(X_exp)

        st.subheader("Global Feature Importance")
        shap.summary_plot(shap_values, X_exp, show=False)
        st.pyplot(plt.gcf())

        st.subheader("Local Explanation (Waterfall)")
        idx = st.slider("Select instance index", min_value=0, max_value=len(X_exp)-1, value=0)
        shap.plots.waterfall(shap_values[idx], show=False)
        st.pyplot(plt.gcf())

    else:
        st.header("5️⃣ Hyperparameter Tuning & Tracking with MLflow")
        df2 = df.dropna().copy()
        features_ht = ["Open", "High", "Low", "Volume"]
        target_ht = "Close"
        X_ht = df2[features_ht]
        y_ht = df2[target_ht]
        X_train_ht, X_test_ht, y_train_ht, y_test_ht = train_test_split(X_ht, y_ht, test_size=0.2, random_state=42)

        est = st.slider("RF n_estimators", 50, 300, 100)
        depth = st.slider("RF max_depth", 2, 20, 5)

        mlflow.set_experiment("MSFT_Hyperparam_Tuning")
        if st.button("Run Experiment"):
            with mlflow.start_run():
                mlflow.log_params({"n_estimators": est, "max_depth": depth})
                ht_model = RandomForestRegressor(n_estimators=est, max_depth=depth, random_state=42)
                ht_model.fit(X_train_ht, y_train_ht)
                preds_ht = ht_model.predict(X_test_ht)
                r2_ht = metrics.r2_score(y_test_ht, preds_ht)
                mlflow.log_metric("r2", r2_ht)
                mlflow.sklearn.log_model(ht_model, "rf_model")
                st.success(f"Experiment logged. R² = {r2_ht:.3f}")

        st.markdown("View your experiment results in the MLflow UI (http://localhost:5000) or on DagsHub.")

if __name__ == "__main__":
    main()