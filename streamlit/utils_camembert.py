import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from prdcodetype2label import prdcodetype2label

model_path_load = 'models/camembert'

# Class order used by camembert (order is numeric)
classes_order = list(np.sort(np.array(list(prdcodetype2label.keys()),dtype="int")))

model = None
tokenizer = None
initialized = False

def init():
    global model,tokenizer,initialized

    model = load_model(model_path_load)
    tokenizer = load_tokenizer(model_path_load)
    initialized = True

@st.cache_resource
def load_model(path):
    return CamembertForSequenceClassification.from_pretrained(path)

@st.cache_resource
def load_tokenizer(path):
    return CamembertTokenizer.from_pretrained(path)

def prepare_text_for_prediction(text, tokenizer):
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    return encodings

def predict(text):
    if not initialized:
        init()

    # Préparer le texte pour la prédiction
    encodings = prepare_text_for_prediction(text, tokenizer)
    # Obtenir l'appareil sur lequel le modèle est chargé
    device = next(model.parameters()).device
    # Déplacer les encodings sur le même appareil que le modèle
    encodings = {k: v.to(device) for k, v in encodings.items()}

    # Convertir les logits en probabilités avec softmax
    with torch.no_grad():  # Ne pas calculer de gradient pour cette opération
        predictions = model(**encodings)
        probabilities = torch.softmax(predictions.logits, dim=1)


    return probabilities.cpu().numpy()[0]  # S'assurer que les probabilités sont aussi sur CPU

