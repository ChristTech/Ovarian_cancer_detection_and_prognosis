import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

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

# Call background image
add_bg_from_local('image.png')

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load("multi_risk_model.sav")

# Cache dataset loading
@st.cache_data
def load_survival_both():
    df = pd.read_csv("csv/SurvivalNormalBoth.csv")
    df = df.drop(columns=["Unnamed: 0", "Cancer.Type"], errors='ignore')
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

# Feature explanations
feature_explanations = {
    "XRCC6": "A gene involved in DNA repair via non-homologous end joining.",
    "BRCA1": "A gene linked to DNA repair and breast/ovarian cancer risk.",
    "Figo.Stage": "Cancer staging (I-IV) indicating tumor progression.",
    "Age.Years": "Patient's age in years.",
    # Expand as needed
}

# Calculate feature ranges and defaults from dataset
feature_ranges = {
    feature: {
        "min": survival_both[feature].min(),
        "max": survival_both[feature].max(),
        "mean": survival_both[feature].mean()
    } for feature in features
}

def tooltip_label(label):
    expl = feature_explanations.get(label)
    return f"{label} ℹ️" if expl else label

st.title("🧬 Ovarian Cancer Prognosis Predictor")

st.markdown("""
This tool predicts **Survival, Cancer Progression, and Recurrence** likelihood based on gene expression and clinical data.  
- Use the **Gene Expression** and **Clinical Parameters** tabs to input patient data.
- Use the **Explore Data** tab to visualize gene expression distributions.
- Check the sidebar to use average dataset values for quick input.
""")

# Sidebar for quick settings
st.sidebar.header("Input Settings")
use_default_values = st.sidebar.checkbox("Use average dataset values", value=True)
show_advanced = st.sidebar.checkbox("Show advanced gene inputs", value=False)

with st.form("patient_form"):
    tabs = st.tabs(["🧪 Gene Expression", "🩺 Clinical Parameters", "📊 Explore Data"])

    # Gene Expression inputs
    gene_input = {}
    with tabs[0]:
        st.caption("Adjust gene expression levels (based on dataset ranges)")
        selected_genes = gene_features[:5] if not show_advanced else gene_features  # Limit to top 5 genes unless advanced mode
        for feature in selected_genes:
            label = tooltip_label(feature)
            range_info = feature_ranges[feature]
            default_value = range_info["mean"] if use_default_values else (range_info["min"] + range_info["max"]) / 2
            gene_input[feature] = st.slider(
                label,
                min_value=float(range_info["min"]),
                max_value=float(range_info["max"]),
                step=0.1,
                value=float(default_value),
                help=f"{feature_explanations.get(feature, 'No description available.')} Range in dataset: {range_info['min']:.2f} to {range_info['max']:.2f}"
            )
        if not show_advanced:
            st.info("Enable 'Show advanced gene inputs' in the sidebar to adjust all gene expression levels.")

    # Clinical inputs
    clinical_input = {}
    with tabs[1]:
        st.caption("Provide clinical attributes")
        for feature in clinical_features:
            label = tooltip_label(feature)
            range_info = feature_ranges[feature]
            if feature == "Figo.Stage":
                # Categorical input for Figo.Stage
                options = ["I", "II", "III", "IV"]
                default_idx = options.index("II") if use_default_values else 0
                selected_stage = st.selectbox(
                    label,
                    options=options,
                    index=default_idx,
                    help=feature_explanations.get(feature, "Select the cancer stage.")
                )
                # Map stages to numerical values (assuming model expects numerical)
                stage_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
                clinical_input[feature] = stage_mapping[selected_stage]
            else:
                default_value = range_info["mean"] if use_default_values else (range_info["min"] + range_info["max"]) / 2
                clinical_input[feature] = st.slider(
                    label,
                    min_value=float(range_info["min"]),
                    max_value=float(range_info["max"]),
                    step=1.0 if feature == "Age.Years" else 0.1,
                    value=float(default_value),
                    help=f"{feature_explanations.get(feature, 'No description available.')} Range in dataset: {range_info['min']:.2f} to {range_info['max']:.2f}"
                )

    # Data exploration tab
    with tabs[2]:
        st.header("Gene Expression Exploration")
        st.subheader("Dataset Preview")
        st.write(survival_data.head())

        gene = st.selectbox("Select a gene to plot:", survival_data.columns[1:-1])
        st.subheader(f"Expression Distribution for {gene}")
        fig, ax = plt.subplots()
        sns.histplot(survival_data[gene], kde=True, ax=ax)
        st.pyplot(fig)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = {**gene_input, **clinical_input}
    # Fill missing gene features (if not all provided) with dataset means
    for feature in features:
        if feature not in input_data:
            input_data[feature] = feature_ranges[feature]["mean"]
    input_df = pd.DataFrame([input_data]).fillna(0)

    # No scaling since model was trained on unscaled data
    prediction = model.predict(input_df)[0]

    st.subheader("🔮 Prediction Results")
    results = {}
    for i, outcome in enumerate(target_labels):
        results[outcome] = prediction[i]
        status = "Likely" if prediction[i] == 1 else "Unlikely"
        emoji = "✅" if prediction[i] == 1 else "❌"
        st.markdown(f"- **{outcome}:** {status} {emoji}")

    # Dynamic advice
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
    advice = advice_dict.get(combo, "Consult a healthcare professional for personalized advice.")
    st.info(advice)

    # Feature importance visualization
    st.subheader("📊 Feature Importance")
    if hasattr(model, 'estimators_'):
        importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [features[i] for i in top_idx]
        top_values = importances[top_idx]
        st.bar_chart(pd.DataFrame({"Importance": top_values}, index=top_features))
    else:
        st.warning("This model does not support feature importance visualization.")