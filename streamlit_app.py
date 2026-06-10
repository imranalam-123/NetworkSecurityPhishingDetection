import streamlit as st
import pandas as pd
import requests
import io

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Network Security Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================================
# Title
# ==========================================================

st.title("🛡️ Network Security Phishing Detection")

# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    # Read uploaded file
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
                df = df.drop(columns=["Result"])

            # Convert dataframe to CSV string
            csv_buffer = io.StringIO()

            df.to_csv(
                csv_buffer,
                index=False
            )

            files = {
                "file": (
                    "uploaded_file.csv",
                    csv_buffer.getvalue(),
                    "text/csv"
                )
            }

            # Call FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                prediction_df = pd.DataFrame(
                    result["predictions"]
                )

                # ==========================================
                # Success Message
                # ==========================================

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
                # Summary Metrics
                # ==========================================

                st.subheader(
                    "Prediction Summary"
                )

                prediction_counts = (
                    prediction_df[
                        "predicted_column"
                    ].value_counts()
                )

                st.write(
                    "Prediction Distribution:"
                )

                st.write(
                    prediction_counts
                )

                phishing_count = prediction_counts.get(
                    0,
                    0
                )

                safe_count = prediction_counts.get(
                    1,
                    0
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
                # Chart
                # ==========================================

                st.subheader(
                    "Prediction Distribution"
                )

                chart_df = pd.DataFrame(
                    {
                        "Category": [
                            "Phishing",
                            "Safe"
                        ],
                        "Count": [
                            phishing_count,
                            safe_count
                        ]
                    }
                )

                st.bar_chart(
                    chart_df.set_index(
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

            else:

                st.error(
                    f"Prediction Failed: {response.text}"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )