from flask import Flask, redirect, render_template, request, jsonify, url_for
import numpy as np
import pickle
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data import IUPACData
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer, AutoModel
import os
import torch
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load pre-trained tokenizer and model for sequence embeddings
tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S")

# Load the trained SVM model
with open("models/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Ensure the upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to calculate binary contact map from PDB files with threshold
def calculate_contact_map(pdb_file, threshold=8.0):
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

    distances = pdist(coords)
    distance_matrix = squareform(distances)

    # Apply threshold to create binary contact map
    contact_map = (distance_matrix <= threshold).astype(int)

    # Pad the contact map to a fixed size
    padded_contact_map = np.zeros((500, 500))
    size = min(contact_map.shape[0], 500)
    padded_contact_map[:size, :size] = contact_map[:size, :size]

    return padded_contact_map.flatten()


# Function to extract sequence from a PDB file
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


# Function to generate sequence embeddings using ESM-1b
def generate_seqvec_embedding(sequence, max_length=1024):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


@app.route('/')
def upload():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'pdb_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdb_file']
    filename = file.filename

    if not (filename.endswith('.pdb') or filename.endswith('.cif')):
        return jsonify({'error': 'Invalid file format. Upload a .pdb or .cif file'}), 400

    pdb_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdb_file_path)

    # Generate features for the uploaded file
    contact_map = calculate_contact_map(pdb_file_path)  # now uses threshold of 8.0
    sequence = extract_sequence_from_pdb(pdb_file_path)
    embedding = generate_seqvec_embedding(sequence)

    # Combine features and make prediction
    features = np.concatenate((contact_map, embedding)).reshape(1, -1)
    probability = svm.predict_proba(features)[0][1]
    diagnosis = 'Positive' if probability >= 0.5 else 'Negative'

    return redirect(url_for('result', probability=probability, diagnosis=diagnosis))


@app.route('/result')
def result():
    probability = request.args.get('probability', None, type=float)
    diagnosis = request.args.get('diagnosis', None)
    return render_template('result.html', probability=probability, diagnosis=diagnosis)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
