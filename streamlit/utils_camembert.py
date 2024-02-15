from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model_path_load = 'models/camembert'

df = pd.read_csv('X_train_prep.csv', index_col=0)

def load_model():
    return CamembertForSequenceClassification.from_pretrained(model_path_load)

def load_tokenizer():
    return CamembertTokenizer.from_pretrained(model_path_load)

def load_encoder():
    return LabelEncoder().fit(df['prdtypecode'])

def prepare_text_for_prediction(text, tokenizer):
    # Tokeniser le texte
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    return encodings

def get_prediction_for_text(text, tokenizer, model, label_encoder):
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
        predicted_category = torch.argmax(probabilities, dim=1)

    # Convertir la catégorie prédite en format approprié pour LabelEncoder
    predicted_category_cpu = predicted_category.cpu().numpy()  # S'assurer que le tensor est sur CPU
    predicted_category_original = label_encoder.inverse_transform(predicted_category_cpu)

    return predicted_category_original, probabilities.cpu().numpy()  # S'assurer que les probabilités sont aussi sur CPU

def predictCamembert(text):
    model = load_model()
    tokenizer = load_tokenizer()
    label_encoder = load_encoder()
    return get_prediction_for_text(text, tokenizer=tokenizer, model=model, label_encoder=label_encoder)