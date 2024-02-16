import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import keras
from prdcodetype2label import prdcodetype2label

# order of class labels returned in model.predict columns
classes_order = list(
	np.sort(np.array(list(prdcodetype2label.keys()),dtype="str")).astype("int")
)

model_path_load = 'models/vgg16.keras'
model = None
initialized = False

def init(): 
	global model,initialized
	model = load_model(model_path_load)
	initialized = True

@keras.saving.register_keras_serializable(name='w_f1_score')
def w_f1_score(y_true, y_pred):
    f1 = tf.py_function(f1_score_sklearn, (y_true, y_pred), tf.float64)
    return f1

@st.cache_resource
def load_model(path):
	return keras.models.load_model(path)


def predict(image,prob_class_order):
	if not initialized:
		init()
	x = preprocess_input(np.expand_dims(image, axis=0) ) 
	pred = model.predict(x)[0] 
	reordered_pred = reorder_predict_cols(pred,classes_order,prob_class_order)
	return reordered_pred


def reorder_predict_cols(pred,old_ordered_classes,new_ordered_classes):
  old_classes_to_index = {}
  for i in range(len(old_ordered_classes)):
    old_classes_to_index[old_ordered_classes[i]] = i

  reordered_indexes = [old_classes_to_index[c] for c in new_ordered_classes]
  reordered_pred = pred[reordered_indexes]
  return reordered_pred