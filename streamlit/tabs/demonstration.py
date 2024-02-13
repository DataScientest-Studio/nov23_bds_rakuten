import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from PIL import Image
from prdcodetype2label import prdcodetype2label
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie 
from utils import load_lottiefile

def renderDemonstration():
    st.title('Démonstration')
    st.divider()

    option = option_menu(None, ['Texte', 'Images', 'Fusion'], 
        icons=['chat-text', "images", "file-richtext"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    with st.container():
        col1,col2=st.columns([2, 5])
        with col1:            
            with st.form('predict_form', clear_on_submit=False):
                option1 = st.selectbox(
                        'Modèle texte : ',
                        ('CamenBERT', 'SVM')
                    )
                option2 = st.selectbox(
                        'Modèle image : ',
                        ('VGG16', 'Resnet152')
                    )
            
                if option in ['Texte', 'Fusion']:
                    designation = st.text_input(
                        "Désignation",
                        placeholder = "Désignation et description du produit"
                    )
                    description = st.text_area(
                        "Description",
                        placeholder = "Description du produit"
                    )

                if option == 'Fusion':
                    st.text_input("Rakuten URL", placeholder = 'Lien vers produit Rakuten')

                if option in ['Images', 'Fusion']:
                    uploaded_file = st.file_uploader("Importer une image", type=['png', 'jpg'])
                    if uploaded_file is not None:
                        image = Image.open(uploaded_file)
                        image.load()
                        image = image.resize((224, 224))
                        img_array = np.asarray(image)
                        uploaded_file = img_array.reshape((1, 224, 224, 3))
                        st.image(uploaded_file)
                
                submitted = st.form_submit_button('Prédire', use_container_width=True)

        with col2:
            if submitted:
                predictions = [1.6963866e-01, 1.6472489e-02, 7.9600923e-03, 1.6750157e-04, 1.7716656e-02,
  1.8811231e-02, 1.8812216e-05, 1.5219175e-05, 1.8788783e-03, 2.5968954e-03,
  3.5805337e-04, 7.9823582e-04, 1.2348393e-05, 2.8465588e-03, 7.3200058e-06,
  6.4861789e-02, 6.0710788e-02, 2.7548616e-05, 9.3458244e-04, 1.6446085e-06,
  2.2978909e-05, 1.4562180e-05, 6.1466205e-01, 1.4588750e-04, 1.9193998e-02,
  1.2492076e-04, 3.0853397e-07] #np.array(models['vgg16'].predict(uploaded_file)[0])
                class_names = [prdcodetype2label[i] for i in prdcodetype2label]

                class_predictions = list(zip(predictions, prdcodetype2label.values()))

                # Trier les prédictions par probabilité (du plus élevé au plus bas)
                sorted_predictions = sorted(class_predictions, key=lambda x: x[0], reverse=True)

                # Garder seulement les cinq meilleures prédictions
                top_5_predictions = sorted_predictions[:5]

                # Extraire les noms de classe et les probabilités
                top_class_names, top_probabilities = zip(*top_5_predictions)

                fig, ax = plt.subplots(figsize=(15, 4))
                ax.bar(class_names, predictions)
                ax.set_title('Prédictions par catégorie', fontsize=16)
                ax.set_ylabel('Probabilités', fontsize=14)
                plt.xticks(rotation=90)
                st.pyplot(fig)

                final_prediction = sorted_predictions[0]
                st.subheader('Prédiction finale:' + final_prediction[1])

            else:
                c1, c2, c3 = st.columns([1, 5, 1])
                with c2:
                    lottie = load_lottiefile('assets/dashboard.json')
                    st_lottie(lottie,key='demo', width=600)