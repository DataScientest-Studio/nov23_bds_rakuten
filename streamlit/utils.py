import streamlit as st
import json
import pandas as pd
import keras

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

@st.cache_data
def pull_clean():
    return pd.read_csv('X_train_prep.csv', index_col=0)

@st.cache_resource
def load_models():
    models = {}
    with open('models/vgg16.keras', 'rb') as file:
        models['vgg16'] = keras.models.load_model(file.name, compile=False)
    return models

def get_average_pred(img_pred,text_pred,img_pred_weight=0.3,text_pred_weight=0.6):
  combined_pred = (img_pred * img_pred_weight) + (text_pred * text_pred_weight)
  return combined_pred[0]