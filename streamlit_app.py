import streamlit as st
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Network Security Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================================
# Load Model Once
# ==========================================================

@st.cache_resource
def load_model():

    preprocessor = load_object(
        "final_model/preprocessor.pkl"
    )

    model = load_object(
        "final_model/model.pkl"
    )

    network_model = NetworkModel(
        preprocessor=preprocessor,
        model=model
    )

    return network_model


network_model = load_model()

# ==========================================================
# Title
# ==========================================================

st.title("🛡️ Network Security Phishing Detection")

st.markdown(
    """
    Upload a CSV file containing website features.
    The model will classify URLs as phishing or safe.
    """
)

# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ======================================================
    # Predict Button
    # ======================================================

    if st.button("Predict"):

        try:

            # Remove target column if present
            if "Result" in df.columns:
                df = df.drop(
                    columns=["Result"]
                )

            # ==========================================
            # Prediction
            # ==========================================

            y_pred = network_model.predict(df)

            prediction_df = df.copy()

            prediction_df[
                "predicted_column"
            ] = y_pred

            st.success(
                "Prediction Completed Successfully"
            )

            # ==========================================
            # Prediction Results
            # ==========================================

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

            # ==========================================
            # Summary
            # ==========================================

            st.subheader(
                "Prediction Summary"
            )

            prediction_counts = (
                prediction_df[
                    "predicted_column"
                ].value_counts()
            )

            phishing_count = (
                prediction_counts.get(0, 0)
            )

            safe_count = (
                prediction_counts.get(1, 0)
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Phishing URLs",
                    phishing_count
                )

            with col2:
                st.metric(
                    "Safe URLs",
                    safe_count
                )

            # ==========================================
            # Distribution Table
            # ==========================================

            distribution_df = pd.DataFrame(
                {
                    "Category": [
                        "Phishing URLs",
                        "Safe URLs"
                    ],
                    "Count": [
                        phishing_count,
                        safe_count
                    ]
                }
            )

            st.subheader(
                "Prediction Distribution"
            )

            st.dataframe(
                distribution_df,
                use_container_width=True
            )

            # ==========================================
            # Chart
            # ==========================================

            st.bar_chart(
                distribution_df.set_index(
                    "Category"
                )
            )

            # ==========================================
            # Download Predictions
            # ==========================================

            csv_download = (
                prediction_df.to_csv(
                    index=False
                )
            )

            st.download_button(
                label="📥 Download Predictions",
                data=csv_download,
                file_name="prediction_output.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )