# ==============================================================
# Streamlit App for Drug-Drug Interaction Prediction
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import snf
import os

# ==============================================================
# Page Setup
# ==============================================================

st.set_page_config(page_title="Drug Interaction Predictor", layout="wide")
st.title("💊 Data-Driven Prediction of Drug-Drug Interactions")
st.markdown("### A Deep Learning-based Clinical Decision Support Tool")

# ==============================================================
# Load Model
# ==============================================================

MODEL_PATH = "final_model.keras"

@st.cache_resource
def load_ddi_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please ensure 'final_model.keras' exists in the project directory.")
        return None
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_ddi_model()

# ==============================================================
# Load Drug Metadata (Dummy Example – Replace with Real Data)
# ==============================================================

@st.cache_data
def load_drug_data():
    # Replace with your actual CSV paths
    base_path = "./data"
    try:
        df_drug_list = pd.read_csv(f"{base_path}/drug_list.csv")  # must contain 'Drug Name' & 'DrugBank ID'
    except FileNotFoundError:
        df_drug_list = pd.DataFrame({
            "Drug Name": ["Aspirin", "Paracetamol", "Ibuprofen"],
            "DrugBank ID": ["DB00945", "DB00316", "DB01050"]
        })
    drug_info_dict = {
        "DB00945": {"Side Effects": "Nausea, bleeding", "Target Sites": "COX enzymes"},
        "DB00316": {"Side Effects": "Liver toxicity (high dose)", "Target Sites": "CNS"},
        "DB01050": {"Side Effects": "Gastrointestinal irritation", "Target Sites": "COX-1/COX-2"}
    }
    drug_names = list(df_drug_list["Drug Name"])
    return df_drug_list, drug_info_dict, drug_names

df_drug_list, drug_info_dict, drug_names = load_drug_data()

# ==============================================================
# Helper Function for Prediction
# ==============================================================

def predict_interaction(drug1, drug2):
    # Mock similarity matrices (replace with actual matrices)
    similarity_matrices = [np.random.rand(3, 3) for _ in range(5)]
    fused_similarity = snf.snf(similarity_matrices, K=20)
    drug_fea = np.random.rand(3, 20)  # Mock drug features

    if drug1 not in drug_names or drug2 not in drug_names:
        st.warning("One or both drug names not found in dataset.")
        return None

    idx1 = drug_names.index(drug1)
    idx2 = drug_names.index(drug2)

    drug1_features = drug_fea[idx1]
    drug2_features = drug_fea[idx2]
    similarities = [sim[idx1, idx2] for sim in similarity_matrices]

    combined_features = np.concatenate([drug1_features.flatten(), drug2_features.flatten(), similarities])
    expected_input_size = model.input_shape[1]
    combined_features = np.pad(combined_features, (0, expected_input_size - combined_features.shape[0]))[:expected_input_size]
    combined_features = combined_features.reshape(1, -1)

    prediction = model.predict(combined_features)
    if prediction.shape[-1] == 2:
        predicted_label = np.argmax(prediction)
        probability = float(prediction[0][1])
    else:
        predicted_label = int(prediction[0] > 0.5)
        probability = float(prediction[0])

    drug1_id = df_drug_list.loc[df_drug_list["Drug Name"] == drug1, "DrugBank ID"].values[0]
    drug2_id = df_drug_list.loc[df_drug_list["Drug Name"] == drug2, "DrugBank ID"].values[0]

    return {
        "Drug 1": drug1,
        "Drug 2": drug2,
        "Interaction Risk": "⚠️ Likely Interaction" if predicted_label == 1 else "✅ Safe Combination",
        "Probability": f"{probability*100:.2f}%",
        "Drug 1 Side Effects": drug_info_dict.get(drug1_id, {}).get("Side Effects", "N/A"),
        "Drug 2 Side Effects": drug_info_dict.get(drug2_id, {}).get("Side Effects", "N/A"),
        "Drug 1 Targets": drug_info_dict.get(drug1_id, {}).get("Target Sites", "N/A"),
        "Drug 2 Targets": drug_info_dict.get(drug2_id, {}).get("Target Sites", "N/A")
    }

# ==============================================================
# Streamlit Layout
# ==============================================================

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("Select First Drug", options=drug_names)
with col2:
    drug2 = st.selectbox("Select Second Drug", options=drug_names)

if st.button("🔍 Predict Interaction"):
    with st.spinner("Analyzing possible interactions..."):
        if model is not None:
            result = predict_interaction(drug1, drug2)
            if result:
                st.success("Prediction complete ✅")
                st.subheader("📊 Prediction Results")
                st.write(f"**{result['Interaction Risk']}** (Probability: {result['Probability']})")

                st.markdown("---")
                st.subheader("Drug Details")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{result['Drug 1']}**")
                    st.write("🧬 Target Sites:", result["Drug 1 Targets"])
                    st.write("⚕️ Side Effects:", result["Drug 1 Side Effects"])
                with c2:
                    st.markdown(f"**{result['Drug 2']}**")
                    st.write("🧬 Target Sites:", result["Drug 2 Targets"])
                    st.write("⚕️ Side Effects:", result["Drug 2 Side Effects"])
        else:
            st.error("Model not loaded. Please check file path.")

st.markdown("---")
st.caption("Developed using Streamlit • Keras • TensorFlow • SNF • Pandas")

