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
    try:
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
    except FileNotFoundError:
        st.warning("Background image not found. Using default background.")

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
    df = df.drop(columns=["Unnamed: 0", "Cancer.Type"], errors='ignore')
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
# Feature Ranges and Explanations
# ===========================
feature_ranges = {
    feature: {
        "min": data[feature].min(),
        "max": data[feature].max(),
        "median": data[feature].median()
    } for feature in features
}

feature_explanations = {
    "XRCC6": "Involved in DNA repair. Normal range: 4-6",
    "BRCA1": "Gene linked to breast/ovarian cancer. Normal range: 5-7",
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
    # Add more readable names as needed
}

# ===========================
# Streamlit App
# ===========================
st.title("🧬 Ovarian Cancer Prognosis Predictor")
st.markdown("""
This tool predicts a patient’s **Survival, Cancer Progression, and Recurrence** likelihood based on gene and clinical data.  
- Use the **Gene Expression** and **Clinical Parameters** tabs to input data.
- Use the **Explore Data** tab to visualize distributions.
- Select **Basic** mode for simplified inputs or **Advanced** for all features.
""")

mode = st.radio("Select Input Mode:", ["Basic (Recommended)", "Advanced"], index=0)

with st.form("patient_form"):
    tabs = st.tabs(["🧪 Gene Expression", "🩺 Clinical Parameters", "📊 Explore Data"])

    # ===========================
    # Gene Expression
    # ===========================
    gene_input = {}
    with tabs[0]:
        st.caption("Adjust gene expression levels (based on dataset ranges)")
        selected_genes = ["XRCC6", "BRCA1"] if mode == "Basic (Recommended)" else gene_features
        for feature in selected_genes:
            label = tooltip_label(feature)
            range_info = feature_ranges[feature]
            gene_input[feature] = st.slider(
                label,
                min_value=float(range_info["min"]),
                max_value=float(range_info["max"]),
                value=float(range_info["median"]),
                step=0.1,
                help=f"{feature_explanations.get(feature, 'No description available.')} Range in dataset: {range_info['min']:.2f} to {range_info['max']:.2f}"
            )

    # ===========================
    # Clinical Parameters
    # ===========================
    clinical_input = {}
    with tabs[1]:
        st.caption("Provide values for clinical attributes")
        selected_features = ["Figo.Stage", "Age.Years"] if mode == "Basic (Recommended)" else clinical_features
        for feature in selected_features:
            label = tooltip_label(rename_dict.get(feature, feature))
            range_info = feature_ranges[feature]
            if feature == "Figo.Stage":
                options = ["I", "II", "III", "IV"]
                default_idx = options.index("II") if range_info["median"] >= 2 else 0
                selected_stage = st.selectbox(
                    label,
                    options=options,
                    index=default_idx,
                    help=feature_explanations.get(feature, "Select the cancer stage.")
                )
                stage_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
                clinical_input[feature] = stage_mapping[selected_stage]
            else:
                clinical_input[feature] = st.slider(
                    label,
                    min_value=float(range_info["min"]),
                    max_value=float(range_info["max"]),
                    value=float(range_info["median"]),
                    step=1.0 if feature == "Age.Years" else 0.1,
                    help=f"{feature_explanations.get(feature, 'No description available.')} Range in dataset: {range_info['min']:.2f} to {range_info['max']:.2f}"
                )

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

    # Submit button outside tabs
    submitted = st.form_submit_button("🔍 Predict")

# ===========================
# Prediction and Results
# ===========================
if submitted:
    input_data = {**gene_input, **clinical_input}
    # Fill missing features with dataset medians
    for feature in features:
        if feature not in input_data:
            input_data[feature] = feature_ranges[feature]["median"]
    input_df = pd.DataFrame([input_data]).fillna(0)

    try:
        prediction = model.predict(input_df)[0]
        # Ensure prediction is numeric
        prediction = [int(p) for p in prediction]

        st.subheader("🔮 Prediction Results")
        results = {}
        for i, outcome in enumerate(target_labels):
            results[outcome] = prediction[i]
            status = "Likely" if prediction[i] == 1 else "Unlikely"
            emoji = "✅" if prediction[i] == 1 else "❌"
            st.markdown(f"- **{outcome}:** {status} {emoji}")

        # Prediction probabilities (if supported)
        if hasattr(model, 'predict_proba'):
            st.markdown("### 🔢 Prediction Probabilities")
            try:
                probas = model.predict_proba(input_df)
                for i, outcome in enumerate(target_labels):
                    st.markdown(f"- **{outcome}:** {probas[i][0][1]*100:.1f}% chance")
            except Exception as e:
                st.warning(f"Could not compute probabilities: {e}")

        # Health Advice
        st.subheader("📋 Health Advice")
        combo = tuple(results.values())
        advice_dict = {
            (1, 0, 0): "Focus on maintaining treatment response; monitor for progression or recurrence.",
            (0, 1, 0): "Consider targeted therapies to manage progression risk.",
            (0, 0, 1): "Regular screening is crucial to detect potential recurrence early.",
            (1, 1, 0): "Close monitoring and possibly aggressive treatment recommended.",
            (1, 0, 1): "Comprehensive follow-up and treatment adjustments are advised.",
            (0, 1, 1): "High risk; consult a multidisciplinary team for management.",
            (1, 1, 1): "Very high risk; immediate comprehensive intervention needed.",
            (0, 0, 0): "No immediate concerns. Continue regular checkups."
        }
        advice = advice_dict.get(combo, "Consult a medical professional for personalized advice.")
        st.info(advice)

        # Feature Importance
        st.subheader("📊 Feature Importance")
        if hasattr(model, 'estimators_'):
            importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
            top_idx = np.argsort(importances)[::-1][:10]
            top_features = [features[i] for i in top_idx]
            top_values = importances[top_idx]
            st.bar_chart(pd.DataFrame({"Importance": top_values}, index=[rename_dict.get(f, f) for f in top_features]))
        else:
            st.warning("This model does not support feature importance visualization.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")