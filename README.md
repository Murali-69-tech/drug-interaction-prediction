# drug-interaction-prediction
Deep learning-based prediction of drug-drug interactions using Keras and Streamlit.


# ==============================================================
# Data-Driven Prediction of Drug Interactions
# Deep Learning-based DDI Prediction using Keras & Streamlit
# ==============================================================

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---- Install Dependencies (run once if needed) ----
# !pip install pandas matplotlib numpy scikit-learn keras tensorflow streamlit snfpy

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_curve, auc, precision_recall_curve
)
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
import snf  # Similarity Network Fusion

# ==============================================================
#  Load Datasets
# ==============================================================

base_path = r"C:\Users\sprag\OneDrive\Attachments\Desktop\project"  # <-- Change this path to your local folder

df_chem_jaccard = pd.read_csv(f"{base_path}\\chem_Jacarrd_sim.csv", index_col=0)
df_atc = pd.read_csv(f"{base_path}\\ATCSimilarityMat.csv", index_col=0)
df_chemical = pd.read_csv(f"{base_path}\\chemicalSimilarityMat.csv", index_col=0)
df_target_jaccard = pd.read_csv(f"{base_path}\\target_Jacarrd_sim.csv", index_col=0)
df_pathway_jaccard = pd.read_csv(f"{base_path}\\pathway_Jacarrd_sim.csv", index_col=0)
df_enzyme_jaccard = pd.read_csv(f"{base_path}\\enzyme_Jacarrd_sim.csv", index_col=0)
df_sideeffect = pd.read_csv(f"{base_path}\\SideEffectSimilarityMat.csv", index_col=0)
df_offsideeffect = pd.read_csv(f"{base_path}\\offsideeffect_Jacarrd_sim.csv", index_col=0)

matrices = [
    df_chem_jaccard.values,
    df_atc.values,
    df_chemical.values,
    df_target_jaccard.values,
    df_pathway_jaccard.values,
    df_enzyme_jaccard.values,
    df_sideeffect.values,
    df_offsideeffect.values
]

# Placeholder dataset dictionary
datasets = {
    "IntegratedDS1": None,  # <-- Replace with actual dataset DataFrame
    "Drug-Drug Matrix": None,
    "DDI Matrix": None,
    "Chemical Jaccard": df_chem_jaccard,
    "ATC Similarity": df_atc,
    "Chemical Similarity": df_chemical,
    "Target Jaccard": df_target_jaccard,
    "Pathway Jaccard": df_pathway_jaccard,
    "Enzyme Jaccard": df_enzyme_jaccard,
    "Side Effect Similarity": df_sideeffect,
    "Offside Effect Jaccard": df_offsideeffect
}

# ==============================================================
#  Prepare Data for Model
# ==============================================================

def prepare_data():
    if "IntegratedDS1" not in datasets or "Drug-Drug Matrix" not in datasets:
        raise ValueError("Error: Required datasets not found!")

    drug_fea = MinMaxScaler().fit_transform(datasets["IntegratedDS1"].values)
    interaction = datasets["Drug-Drug Matrix"].values

    fused_similarity = snf.snf(matrices, K=20)  # Fuse similarity matrices

    train, label = [], []
    min_size = min(drug_fea.shape[0], fused_similarity.shape[0])

    for i in range(min_size):
        for j in range(min_size):
            if i != j:
                combined_features = list(drug_fea[i]) + list(drug_fea[j]) + [fused_similarity[i, j]]
                train.append(combined_features)
                label.append(int(interaction[i, j]))

    train = np.array(train)
    label = np.array(label)
    label = to_categorical(label, num_classes=2)

    print("Train feature vector shape:", train.shape)
    print("Label shape:", label.shape)

    return train, label, drug_fea, fused_similarity


# ==============================================================
#  Prediction Function
# ==============================================================

def predict_interaction(drug1_name, drug2_name, drug_fea, model, similarity_matrices, drug_names, drug_info_dict, df_drug_list):
    if drug1_name not in drug_names or drug2_name not in drug_names:
        return {"Error": "One or both drugs not found!"}

    idx1 = drug_names.index(drug1_name)
    idx2 = drug_names.index(drug2_name)

    drug1_features = drug_fea[idx1]
    drug2_features = drug_fea[idx2]
    similarities = [sim[idx1, idx2] for sim in similarity_matrices]

    combined_features = np.concatenate([drug1_features.flatten(), drug2_features.flatten(), similarities])
    expected_input_size = model.input_shape[1]

    if combined_features.shape[0] > expected_input_size:
        combined_features = combined_features[:expected_input_size]
    else:
        combined_features = np.pad(combined_features, (0, expected_input_size - combined_features.shape[0]))

    combined_features = combined_features.reshape(1, -1)
    prediction = model.predict(combined_features)

    if prediction.shape[-1] == 2:
        predicted_label = np.argmax(prediction)
        probability = float(prediction[0][1])
    else:
        predicted_label = int(prediction[0] > 0.5)
        probability = float(prediction[0])

    drug1_id = df_drug_list.loc[df_drug_list["Drug Name"] == drug1_name, "DrugBank ID"].values
    drug2_id = df_drug_list.loc[df_drug_list["Drug Name"] == drug2_name, "DrugBank ID"].values

    drug1_id = drug1_id[0] if len(drug1_id) > 0 else "N/A"
    drug2_id = drug2_id[0] if len(drug2_id) > 0 else "N/A"

    drug1_info = drug_info_dict.get(drug1_id, {"Side Effects": "N/A", "Target Sites": "N/A"})
    drug2_info = drug_info_dict.get(drug2_id, {"Side Effects": "N/A", "Target Sites": "N/A"})

    return {
        "Drug 1": drug1_name,
        "Drug 2": drug2_name,
        "Predicted Label": predicted_label,
        "Probability": probability,
        "Drug 1 Side Effects": drug1_info["Side Effects"],
        "Drug 2 Side Effects": drug2_info["Side Effects"],
        "Drug 1 Target Sites": drug1_info["Target Sites"],
        "Drug 2 Target Sites": drug2_info["Target Sites"]
    }

# ==============================================================
#  Load Model
# ==============================================================

model_path = "final_model.keras"  # Path to trained Keras model
if not os.path.exists(model_path):
    print("⚠️ Model file not found. Please ensure 'final_model.keras' is available.")
else:
    model = load_model(model_path, compile=False)
    print("✅ Model Loaded Successfully!")
    model.summary()


# ==============================================================
#  Example Usage
# ==============================================================

# (You should replace the below variables with your actual data)
drug_names = []          # e.g., list of drug names
drug_info_dict = {}      # e.g., { "DB0001": {"Side Effects": "...", "Target Sites": "..."} }
df_drug_list = pd.DataFrame(columns=["Drug Name", "DrugBank ID"])
similarity_matrices = matrices
drug_fea = np.random.rand(10, 20)  # dummy placeholder

# Example Prediction
# result = predict_interaction("Aspirin", "Paracetamol", drug_fea, model, similarity_matrices, drug_names, drug_info_dict, df_drug_list)
# print(result)

# ==============================================================
# End of Script
# ==============================================================

