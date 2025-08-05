import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call like:
add_bg_from_local('image.png')

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load("multi_risk_model.sav")

# Cache dataset loading
@st.cache_data
def load_survival_both():
    df = pd.read_csv("csv/SurvivalNormalBoth.csv")
    df = df.drop(columns=["Unnamed: 0", "Cancer.Type"])
    return df

@st.cache_data
def load_survival_data():
    return pd.read_csv('csv/SurvivalNormal.csv')

# Load model and data
model = load_model()
survival_both = load_survival_both()
survival_data = load_survival_data()

# Features and targets
features = survival_both.drop(columns=["SURVIVAL", "PROGRESSION", "RECURRENCE"]).columns.tolist()
target_labels = ["Survival", "Progression", "Recurrence"]

gene_features = [f for f in features if f.isupper()]
clinical_features = [f for f in features if f not in gene_features]

# Simple explanations dictionary (expand as needed)
feature_explanations = {
    "XRCC6": "A gene involved in DNA repair via non-homologous end joining.",
    "BRCA1": "A gene linked to DNA repair and breast/ovarian cancer risk.",
    "Figo.Stage": "Cancer staging system indicating tumor progression.",
    "Age.Years": "Patient's age in years.",
    # Add more here...
}

def tooltip_label(label):
    expl = feature_explanations.get(label)
    if expl:
        return f"{label} ℹ️"
    else:
        return label

st.title("🧬 Ovarian Cancer Prognosis Predictor")

st.markdown("""
This tool predicts a patient’s **Survival, Cancer Progression, and Recurrence** likelihood based on gene and clinical data.

Use the **Explore Data** tab to view gene expression distributions.
""")

with st.form("patient_form"):

    tabs = st.tabs(["🧪 Gene Expression", "🩺 Clinical Parameters", "📊 Explore Data"])

    # Gene Expression inputs
    gene_input = {}
    with tabs[0]:
        st.caption("Adjust gene expression levels (normalized values)")
        for feature in gene_features:
            label = tooltip_label(feature)
            gene_input[feature] = st.slider(label, min_value=0.0, max_value=10.0, step=0.1, value=5.0, help=feature_explanations.get(feature))

    # Clinical inputs
    clinical_input = {}
    with tabs[1]:
        st.caption("Provide values for clinical attributes")
        for feature in clinical_features:
            label = tooltip_label(feature)
            clinical_input[feature] = st.slider(label, min_value=0.0, max_value=100.0, step=1.0, value=50.0, help=feature_explanations.get(feature))

    # Data exploration tab (no form inputs here)
    with tabs[2]:
        st.header("Gene Expression Exploration")

        st.subheader("Dataset preview")
        st.write(survival_data.head())

        gene = st.selectbox("Select a gene to plot:", survival_data.columns[1:-1])

        st.subheader(f"Expression distribution for {gene}")
        fig, ax = plt.subplots()
        sns.histplot(survival_data[gene], kde=True, ax=ax)
        st.pyplot(fig)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = {**gene_input, **clinical_input}
    input_df = pd.DataFrame([input_data]).fillna(0)

    prediction = model.predict(input_df)[0]

    st.write("Raw Prediction:", prediction)  # Add this line for debugging

    st.subheader("🔮 Prediction Results")
    results = {}
    for i, outcome in enumerate(target_labels):
        results[outcome] = int(prediction[i])  # Convert to integer
        status = "Likely" if results[outcome] == 1 else "Unlikely"
        emoji = "✅" if results[outcome] == 1 else "❌"
        st.markdown(f"- **{outcome}:** {status} {emoji}")

    # Dynamic advice per prediction combo
    st.subheader("📋 Health Advice")

    combo = tuple(results.values())

    advice_dict = {
        (1, 0, 0): "Focus on treatment response and monitor for progression or recurrence.",
        (0, 1, 0): "Consider more aggressive or targeted therapy due to progression risk.",
        (0, 0, 1): "Maintain follow-up screening to detect possible recurrence early.",
        (1, 1, 0): "Close monitoring and aggressive treatment recommended.",
        (1, 0, 1): "Ensure comprehensive follow-up and treatment adjustments.",
        (0, 1, 1): "High risk; multidisciplinary management advised.",
        (1, 1, 1): "Very high risk; immediate and comprehensive intervention needed.",
        (0, 0, 0): "No immediate concerns based on current data. Maintain regular checkups."
    }

    advice = advice_dict.get(combo, "Please consult a healthcare professional for personalized advice.")
    st.info(advice)

    # Feature importance visualization
    st.subheader("📊 Feature Importance (Model-based)")
    if hasattr(model, 'estimators_'):
        importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [features[i] for i in top_idx]
        top_values = importances[top_idx]

        st.bar_chart(pd.DataFrame({"Importance": top_values}, index=top_features))
    else:
        st.warning("This model does not support feature importances.")
