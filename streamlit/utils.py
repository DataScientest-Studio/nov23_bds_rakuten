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


def get_average_pred(img_pred,text_pred, text_pred_weight =  0.5):
  img_pred_weight = 1 - text_pred_weight
  combined_pred = (img_pred * img_pred_weight) + (text_pred * text_pred_weight)
  return combined_pred