import streamlit as st
import json
import pandas as pd
from keras.models import load_model

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

@st.cache_data
def pull_clean():
    return pd.read_csv('X_train_prep.csv', index_col=0)

@st.cache_resource
def load_models():
    model = {}
    with open('models/vgg16.keras', 'rb') as file:
        model['vgg16'] = load_model(file.name, compile=False)
    return model