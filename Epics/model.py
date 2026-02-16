import os
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data import IUPACData
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import pdist, squareform
import pickle

# Ensure necessary directories exist
os.makedirs('data/Healthy_proteins', exist_ok=True)
os.makedirs('data/Unhealthy_proteins', exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load the SeqVec (ESM-1b) model for sequence embedding
tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S", cache_dir=None)
model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S", cache_dir=None)

FIXED_SIZE = 500
THRESHOLD = 8.0  # Threshold in Angstroms for contact map

# Function to calculate contact maps from PDB files with threshold
def calculate_contact_map(pdb_file):
    if pdb_file.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif pdb_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format! Use .pdb or .cif")
    structure = parser.get_structure('protein', pdb_file)
    coords = [
        residue['CA'].get_coord()
        for model in structure for chain in model for residue in chain
        if residue.has_id('CA')
    ]
    coords = np.array(coords)
    if coords.shape[0] < 2:
        print(f"Error: Not enough CA atoms in {pdb_file}")
        return np.zeros((FIXED_SIZE, FIXED_SIZE))
    distances = squareform(pdist(coords))
    contact_map = (distances <= THRESHOLD).astype(int)
    padded_map = np.zeros((FIXED_SIZE, FIXED_SIZE))
    size = min(contact_map.shape[0], FIXED_SIZE)
    padded_map[:size, :size] = contact_map[:size, :size]
    return padded_map

# Function to generate SeqVec embeddings for a protein sequence
def generate_seqvec_embedding(sequence, max_length=1024):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned_sequence = "".join([aa for aa in sequence if aa in valid_amino_acids])
    inputs = tokenizer(cleaned_sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Function to extract a sequence from a PDB file
def extract_sequence_from_pdb(pdb_file):
    if pdb_file.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif pdb_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format! Use .pdb or .cif")
    structure = parser.get_structure('protein', pdb_file)
    sequence = "".join([
        IUPACData.protein_letters_3to1.get(residue.get_resname().capitalize(), 'X')
        for model in structure for chain in model for residue in chain
        if PDB.is_aa(residue, standard=True)
    ])
    return sequence

# Paths to protein directories
healthy_proteins = "data/Healthy_proteins"
unhealthy_proteins = "data/Unhealthy_proteins"

# List PDB files in a directory
def list_files_in_directory(directory_path):
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(".pdb")
    ]

# Prepare data for training
X_contact_maps, X_embeddings, y = [], [], []

# Process healthy proteins
for pdb_file in list_files_in_directory(healthy_proteins):
    contact_map = calculate_contact_map(pdb_file)
    sequence = extract_sequence_from_pdb(pdb_file)
    embedding = generate_seqvec_embedding(sequence)
    X_contact_maps.append(contact_map.flatten())
    X_embeddings.append(embedding)
    y.append(0)

# Process unhealthy proteins
for pdb_file in list_files_in_directory(unhealthy_proteins):
    contact_map = calculate_contact_map(pdb_file)
    sequence = extract_sequence_from_pdb(pdb_file)
    embedding = generate_seqvec_embedding(sequence)
    X_contact_maps.append(contact_map.flatten())
    X_embeddings.append(embedding)
    y.append(1)

# Convert lists to arrays
X_contact_maps = np.array(X_contact_maps)
X_embeddings = np.array(X_embeddings)
y = np.array(y)

# Combine features and train
X = np.concatenate((X_contact_maps, X_embeddings), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("Model training complete. Saved to 'models/svm_model.pkl'.")
