import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

import streamlit as st
import streamlit_authenticator as stauth
import importlib.metadata

# --- Detect streamlit-authenticator version ---
try:
    version_str = importlib.metadata.version("streamlit-authenticator")
    version = tuple(map(int, version_str.split(".")))
except Exception:
    version = (0, 0, 0)  # fallback if version can't be detected

# --- User Authentication Config ---
config = {
    'credentials': {
        'usernames': {
            'user1': {
                'email': 'user1@email.com',
                'name': 'User One',
                'password': '12345'   # ⚠️ hash in production
            },
            'user2': {
                'email': 'user2@email.com',
                'name': 'User Two',
                'password': '67890'
            }
        }
    },
    'cookie': {
        'name': 'credit_risk_cookie',
        'key': 'signature_key123',
        'expiry_days': 30
    },
    'preauthorized': {
        'emails': []
    }
}

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Handle API differences ---
if version < (0, 3, 0):
    # Old API: returns (name, authentication_status, username)
    name, authentication_status, username = authenticator.login("Login", location="sidebar")

elif (0, 3, 0) <= version < (0, 4, 0):
    # Mid API: returns (name, authentication_status)
    name, authentication_status = authenticator.login("Login", location="sidebar")
    username = name

else:
    # New API (>=0.4.0): no location argument, only returns authentication_status
    with st.sidebar:
        authentication_status = authenticator.login("Login")
    if authentication_status:
        user_info = authenticator.get_user_info()
        name = user_info.get("name")
        username = user_info.get("username")
    else:
        name, username = None, None

# --- Authentication UI ---
if authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your email and password")
elif authentication_status:
    st.success(f"Welcome {name}!")
    if version < (0, 4, 0):
        authenticator.logout("Logout", location="sidebar")
    else:
        with st.sidebar:
            authenticator.logout("Logout")



# -------------------- AI ASSISTANT FUNCTION --------------------
def ai_assistant(pred, prob, shap_values, lime_exp, applicant_aligned):
    explanation_text = ""

    # 1. Prediction summary
    if pred[0] == 1:
        explanation_text += f"⚠️ The model predicts this applicant is at HIGH RISK of default with probability {prob[0]:.2f}.\n\n"
    else:
        explanation_text += f"✅ The model predicts this applicant is at LOW RISK of default with probability {prob[0]:.2f}.\n\n"

    # 2. SHAP Insights
    explanation_text += "📊 **SHAP Insights:**\n"
    important_features = shap_values.values[0].argsort()[-3:][::-1]  # top 3
    for i in important_features:
        feature = applicant_aligned.columns[i]
        value = applicant_aligned.iloc[0, i]
        shap_val = shap_values.values[0][i]
        explanation_text += f"- {feature} = {value} contributed {'positively' if shap_val > 0 else 'negatively'} to the risk score.\n"

    # 3. LIME Insights
    explanation_text += "\n📌 **LIME Explanation:**\n"
    for feature, weight in lime_exp.as_list(label=1):
        explanation_text += f"- {feature} with weight {weight:.2f}\n"

    # 4. Policy & Loan Advice
    explanation_text += "\n💡 **Policy Recommendations:**\n"
    if pred[0] == 0:  # Non-default
        explanation_text += "- The applicant qualifies for a personal loan.\n"
        if prob[0] < 0.3:
            explanation_text += "- Recommended: Higher loan amount with longer repayment period (24–36 months).\n"
        elif prob[0] < 0.6:
            explanation_text += "- Recommended: Medium loan amount with repayment period of 12–24 months.\n"
        else:
            explanation_text += "- Recommended: Lower loan amount with strict monitoring and shorter repayment period (6–12 months).\n"
    else:
        explanation_text += "- The applicant should be carefully monitored or rejected due to high risk of default.\n"
        explanation_text += "- Recommend financial literacy training or credit repair program before loan approval.\n"

    return explanation_text


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

applicant_data = None
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

# -------------------- PREDICT BUTTON --------------------
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

        # ✅ SHAP Explanation
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind", axis=1)))
        shap_values = explainer(scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, applicant_aligned, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig)

        # ✅ LIME Explanation
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

        # ✅ AI Assistant Advice
        st.subheader("🤖 AI Assistant Advice")
        assistant_text = ai_assistant(pred, prob, shap_values, explanation, applicant_aligned)
        st.write(assistant_text)

    else:
        st.warning("Please provide applicant data (manual entry or CSV).")

# -------------------- RETRAIN OPTION --------------------
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



















