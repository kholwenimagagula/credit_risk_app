import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import io

# --- Load Model and Data ---
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("credit_risk_dataset.csv")
feature_names = joblib.load("features.pkl")

# --- SHAP Global Importance: Top 10 Features ---
X = df.drop("default_ind", axis=1)
explainer = shap.Explainer(mlp_model.predict, scaler.transform(X))
shap_values_global = explainer(scaler.transform(X))
shap_mean_importance = np.abs(shap_values_global.values).mean(axis=0)
top_features = X.columns[np.argsort(shap_mean_importance)[-10:][::-1]].tolist()

st.title("📊 Credit Risk Assessment Tool (Enhanced)")

# Initialize session state
if "applicants" not in st.session_state:
    st.session_state["applicants"] = []

# --- Applicant Form for Top 10 Features ---
st.sidebar.header("Enter Applicant Data (Top 10 Features)")
applicant_input = {}

for feat in top_features:
    if pd.api.types.is_numeric_dtype(df[feat]):
        applicant_input[feat] = st.sidebar.number_input(f"{feat}", 
                                                        min_value=float(df[feat].min()), 
                                                        max_value=float(df[feat].max()), 
                                                        value=float(df[feat].median()))
    else:
        applicant_input[feat] = st.sidebar.selectbox(f"{feat}", df[feat].unique())

# Add applicant to session
if st.sidebar.button("Add Applicant"):
    st.session_state["applicants"].append(applicant_input.copy())
    st.sidebar.success(f"Applicant #{len(st.session_state['applicants'])} added!")

# --- Run Predictions when at least 3 applicants ---
if len(st.session_state["applicants"]) >= 3:
    if st.sidebar.button("Run Predictions"):
        applicants_df = pd.DataFrame(st.session_state["applicants"])

        # Align and scale
        applicant_aligned = pd.DataFrame(columns=feature_names)
        for i, row in applicants_df.iterrows():
            applicant_aligned.loc[i] = 0
            for col in row.index:
                if col in applicant_aligned.columns:
                    applicant_aligned.loc[i, col] = row[col]

        scaled = scaler.transform(applicant_aligned)

        # Predictions
        prob = mlp_model.predict_proba(scaled)[:, 1]
        pred = mlp_model.predict(scaled)

        results = []
        for i in range(len(prob)):
            if prob[i] > 0.7:
                risk = "High Risk"
            elif prob[i] > 0.4:
                risk = "Medium Risk"
            else:
                risk = "Low Risk"
            results.append({
                "Applicant": i+1,
                "Prediction": "Default" if pred[i] == 1 else "Non-Default",
                "Probability of Default": prob[i],
                "Risk Category": risk
            })

        results_df = pd.DataFrame(results)
        st.subheader("📋 Applicant Comparison")
        st.write(results_df)

        # --- Risk Distribution Chart ---
        st.subheader("📊 Risk Distribution")
        fig, ax = plt.subplots()
        results_df["Risk Category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

        # --- Loan Simulator ---
        st.subheader("⚡ Loan Simulator (What-If Analysis)")
        applicant_choice = st.selectbox("Choose applicant to simulate:", results_df["Applicant"])
        new_loan = st.slider("Adjust Loan Amount", 
                             min_value=1000, max_value=50000, step=5000)

        sim_data = applicant_aligned.loc[applicant_choice-1].copy()
        if "loan_amount" in sim_data.index:
            sim_data["loan_amount"] = new_loan
        sim_scaled = scaler.transform([sim_data])
        sim_prob = mlp_model.predict_proba(sim_scaled)[:, 1][0]
        st.write(f"New predicted default probability: **{sim_prob:.2f}**")

        # --- SHAP Global Feature Importance ---
        st.subheader("🔍 Global Feature Importance")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values_global, X, feature_names=X.columns, plot_type="bar", show=False)
        st.pyplot(fig)

        # --- Export Option ---
        buffer = io.BytesIO()
        results_df.to_csv(buffer, index=False)
        st.download_button("⬇️ Download Results CSV", buffer, "credit_risk_results.csv", "text/csv")

        # --- Applicant History ---
        st.subheader("🕒 Applicant History")
        st.write(pd.DataFrame(st.session_state["applicants"]))
else:
    st.info("Add at least 3 applicants to run predictions.")























