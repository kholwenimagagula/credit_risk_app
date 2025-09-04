import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import os

# Load model, scaler, dataset, and feature names
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("credit_risk_dataset.csv")
feature_names = joblib.load("features.pkl")

st.title("📊 Credit Risk Assessment Tool")

# Sidebar options
st.sidebar.header("Applicant Data Input")

# Choose input mode
input_mode = st.sidebar.radio("Choose input mode:", ["Manual Entry", "Upload CSV"])

if input_mode == "Manual Entry":
    income = st.sidebar.number_input("Income", min_value=0.0, step=100.0)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, step=100.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, step=1)

    applicant_data = pd.DataFrame({
        "income": [income],
        "age": [age],
        "loan_amount": [loan_amount],
        "credit_score": [credit_score]
    })

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Applicant CSV", type=["csv"])
    if uploaded_file:
        applicant_data = pd.read_csv(uploaded_file)
    else:
        applicant_data = None
        
    # Align applicant data to training features
applicant_aligned = pd.DataFrame(columns=feature_names)
applicant_aligned.loc[0] = 0  # initialize with zeros
for col in applicant_data.columns:
    if col in applicant_aligned.columns:
        applicant_aligned.loc[0, col] = applicant_data[col].values[0]

# Predict button
if st.sidebar.button("Predict"):
    if applicant_data is not None:
        # Align applicant data with training features
        applicant_aligned = pd.DataFrame(columns=feature_names)
        applicant_aligned.loc[0] = 0
        for col in applicant_data.columns:
            if col in applicant_aligned.columns:
                applicant_aligned.loc[0, col] = applicant_data[col].values[0]

        # Scale aligned data
        scaled = scaler.transform(applicant_aligned)

        # Prediction
        prob = mlp_model.predict_proba(scaled)[:, 1]
        pred = mlp_model.predict(scaled)

        # Results
        results = []
        for i in range(len(prob)):
            if prob[i] > 0.7:
                risk = "High Risk"
            elif prob[i] > 0.4:
                risk = "Medium Risk"
            else:
                risk = "Low Risk"
            results.append({
                "Prediction": "Default" if pred[i] == 1 else "Non-Default",
                "Probability of Default": prob[i],
                "Risk Category": risk
            })

        results_df = pd.DataFrame(results)
        st.subheader("Prediction Results")
        st.write(results_df)

        # ✅ SHAP Explanation (now inside the same block)
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind", axis=1)))
        shap_values = explainer(scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, applicant_aligned, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig)

        # ✅ LIME Explanation (also inside the block)
        st.subheader("LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(df.drop("default_ind", axis=1).values),
            feature_names=df.drop("default_ind", axis=1).columns.tolist(),
            class_names=["Non-Default", "Default"],
            mode="classification"
        )
        explanation = lime_explainer.explain_instance(
            data_row=scaled[0],
            predict_fn=mlp_model.predict_proba,
            num_features=4
        )
        fig = explanation.as_pyplot_figure(label=1)
        st.pyplot(fig)
    else:
        st.warning("Please provide applicant data (manual entry or CSV).")


        # SHAP explanation
st.subheader("SHAP Explanation")

explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind", axis=1)))
shap_values = explainer(scaled)

# Plot SHAP
st.subheader("SHAP Explanation")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, applicant_aligned, feature_names=feature_names, plot_type="bar", show=False)
st.pyplot(fig)

# Use matplotlib backend for Streamlit
fig, ax = plt.subplots()
shap.summary_plot(shap_values, applicant_data, feature_names=applicant_data.columns, plot_type="bar", show=False)
st.pyplot(fig)


       # LIME explanation
st.subheader("LIME Explanation")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(df.drop("default_ind", axis=1).values),
    feature_names=df.drop("default_ind", axis=1).columns.tolist(),
    class_names=["Non-Default", "Default"],
    mode="classification"
)

explanation = lime_explainer.explain_instance(
    data_row=scaled[0],
    predict_fn=mlp_model.predict_proba,
    num_features=4
)

# Convert to matplotlib figure for Streamlit
fig = explanation.as_pyplot_figure(label=1)
st.pyplot(fig)

# Retrain option
if st.sidebar.button("Retrain Model"):
    from sklearn.neural_network import MLPClassifier

    X = df.drop("default_ind", axis=1)
    y = df["default_ind"]

    scaler.fit(X)
    X_scaled = scaler.transform(X)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp_model.fit(X_scaled, y)

    joblib.dump(mlp_model, "mlp_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("Model retrained successfully with updated dataset!")





