import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# ===========================
# Background Image
# ===========================
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

add_bg_from_local('image.png')

# ===========================
# Cached Resources
# ===========================
@st.cache_resource
def load_model():
    return joblib.load("multi_risk_model.sav")

@st.cache_data
def load_data():
    df = pd.read_csv("csv/SurvivalNormalBoth.csv")
    df = df.drop(columns=["Unnamed: 0", "Cancer.Type"])
    return df

@st.cache_data
def load_data_preview():
    return pd.read_csv("csv/SurvivalNormal.csv")

# ===========================
# Load
# ===========================
model = load_model()
data = load_data()
data_preview = load_data_preview()
features = data.drop(columns=["SURVIVAL", "PROGRESSION", "RECURRENCE"]).columns.tolist()
target_labels = ["Survival", "Progression", "Recurrence"]

gene_features = [f for f in features if f.isupper()]
clinical_features = [f for f in features if f not in gene_features]

# ===========================
# Feature Explanations
# ===========================
feature_explanations = {
    "XRCC6": "Involved in DNA repair. Normal: 4-6",
    "BRCA1": "Gene linked to breast/ovarian cancer. Normal: 5-7",
    "Figo.Stage": "FIGO staging system: 1 (Early) to 4 (Late)",
    "Age.Years": "Patient's age in years",
    # Add more features as needed
}

def tooltip_label(label):
    expl = feature_explanations.get(label)
    return f"{label} ℹ️" if expl else label

rename_dict = {
    "Figo.Stage": "FIGO Stage",
    "Age.Years": "Age (Years)",
    # Add readable names here
}

# ===========================
# Streamlit App
# ===========================
st.title("🧬 Ovarian Cancer Prognosis Predictor")
st.markdown("""
This tool predicts a patient’s **Survival, Cancer Progression, and Recurrence** likelihood based on gene and clinical data. Use the Explore tab for insight.
""")

mode = st.radio("Select Input Mode:", ["Basic (Recommended)", "Advanced"])

with st.form("patient_form"):

    tabs = st.tabs(["🧪 Gene Expression", "🩺 Clinical Parameters", "📊 Explore Data"])

    default_vals = data.median()

    # ===========================
    # Gene Expression
    # ===========================
    gene_input = {}
    with tabs[0]:
        st.caption("Adjust gene expression levels (normalized values)")
        if mode == "Basic (Recommended)":
            for feature in ["XRCC6", "BRCA1"]:
                label = tooltip_label(feature)
                gene_input[feature] = st.slider(label, 0.0, 10.0, float(default_vals[feature]), 0.1, help=feature_explanations.get(feature))
        else:
            for feature in gene_features:
                label = tooltip_label(feature)
                gene_input[feature] = st.slider(label, 0.0, 10.0, float(default_vals[feature]), 0.1, help=feature_explanations.get(feature))

    # ===========================
    # Clinical Parameters
    # ===========================
    clinical_input = {}
    with tabs[1]:
        st.caption("Provide values for clinical attributes")
        selected_features = ["Figo.Stage", "Age.Years"] if mode == "Basic (Recommended)" else clinical_features
        for feature in selected_features:
            label = tooltip_label(rename_dict.get(feature, feature))
            clinical_input[feature] = st.slider(label, 0.0, 100.0, float(default_vals[feature]), 1.0, help=feature_explanations.get(feature))

    # ===========================
    # Data Exploration
    # ===========================
    with tabs[2]:
        st.header("Gene Expression Exploration")
        st.write(data_preview.head())

        gene = st.selectbox("Select a gene to visualize:", data_preview.columns[1:-1])
        fig, ax = plt.subplots()
        sns.histplot(data_preview[gene], kde=True, ax=ax)
        st.pyplot(fig)

    submitted = st.form_submit_button("🔍 Predict")

# ===========================
# Prediction and Results
# ===========================
if submitted:
    input_data = {**gene_input, **clinical_input}
    input_df = pd.DataFrame([input_data]).fillna(0)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("🔮 Prediction Results")
        results = {}
        for i, outcome in enumerate(target_labels):
            results[outcome] = prediction[i]
            status = "Likely" if prediction[i] == 1 else "Unlikely"
            emoji = "✅" if prediction[i] == 1 else "❌"
            st.markdown(f"- **{outcome}:** {status} {emoji}")

        # Optional: Show predicted probabilities if supported
        if hasattr(model, 'predict_proba'):
            st.markdown("### 🔢 Prediction Probabilities")
            try:
                probas = model.predict_proba(input_df)
                for i, outcome in enumerate(target_labels):
                    st.markdown(f"- **{outcome}:** {probas[i][0][1]*100:.1f}% chance")
            except:
                pass

        # Health Advice
        combo = tuple(results.values())
        advice_dict = {
            (1, 0, 0): "Focus on treatment response and monitor for progression or recurrence.",
            (0, 1, 0): "Consider more aggressive or targeted therapy.",
            (0, 0, 1): "Maintain follow-up screening.",
            (1, 1, 0): "Close monitoring and aggressive treatment.",
            (1, 0, 1): "Ensure comprehensive follow-up.",
            (0, 1, 1): "High risk; multidisciplinary care advised.",
            (1, 1, 1): "Very high risk; immediate intervention needed.",
            (0, 0, 0): "No immediate concern. Continue checkups."
        }
        st.subheader("📋 Health Advice")
        advice = advice_dict.get(combo, "Consult a medical professional.")
        st.info(advice)

        # Feature Importance (if supported)
        st.subheader("📊 Feature Importance")
        if hasattr(model, 'estimators_'):
            importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
            top_idx = np.argsort(importances)[::-1][:10]
            top_features = [features[i] for i in top_idx]
            top_values = importances[top_idx]
            st.bar_chart(pd.DataFrame({"Importance": top_values}, index=top_features))
        else:
            st.warning("This model does not support feature importances.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

