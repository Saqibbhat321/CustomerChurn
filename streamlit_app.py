import streamlit as st
import requests


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


API_URL = "http://127.0.0.1:8000/predict"
# Later for Render:
# API_URL = "https://your-fastapi-service.onrender.com/predict"

st.title("📉 Customer Churn Prediction System")
st.markdown(
    """
Predict whether a telecom customer is likely to **churn** or **stay** using a trained **Random Forest model**.
Fill in the customer details below and click **Predict Churn**.
"""
)

st.divider()


with st.form("churn_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    col4, col5 = st.columns(2)

    with col4:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col5:
        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=0.1
        )
        TotalCharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=0.1
        )

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        with st.spinner("Getting prediction from model..."):
            response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            prediction_label = result.get("prediction_label", "Unknown")
            churn_probability = result.get("churn_probability", 0.0)
            confidence = result.get("confidence", "Unknown")
            message = result.get("message", "No message returned.")
            model_name = result.get("model", "Random Forest Classifier")
            api_version = result.get("api_version", "N/A")

            st.divider()
            st.subheader("Prediction Result")

            if prediction_label.lower() == "churn":
                st.error(f"⚠️ Prediction: {prediction_label}")
            else:
                st.success(f"✅ Prediction: {prediction_label}")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")

            with metric_col2:
                st.metric("Confidence", confidence)

            with metric_col3:
                st.metric("Model", model_name)

            st.info(message)

            with st.expander("View API Response"):
                st.json(result)

            with st.expander("View Input Payload"):
                st.json(payload)

            st.caption(f"API Version: {api_version}")

        else:
            st.error(f"API Error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI backend. Make sure your FastAPI server is running on http://127.0.0.1:8000"
        )
    except Exception as e:
        st.error(f"Unexpected error: {e}")