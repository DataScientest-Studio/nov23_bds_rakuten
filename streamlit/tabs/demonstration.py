import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from PIL import Image
from prdcodetype2label import prdcodetype2label
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie 
from utils import load_lottiefile, get_average_pred
import utils_camembert
import utils_vgg16
from scrapper import scrap
import requests
import numpy as np
import pandas as pd


if 'designation_input' not in st.session_state:
    st.session_state['designation_input'] = ''
if 'description_input' not in st.session_state:
    st.session_state['description_input'] = ''
if 'class_input' not in st.session_state:
    st.session_state['class_input'] = ''
if 'image_input' not in st.session_state:
    st.session_state['image_input'] = ''
if 'scrap_input' not in st.session_state:
    st.session_state['scrap_input'] = ''
if 'image_url' not in st.session_state:
    st.session_state['image_url'] = ''

@st.cache_data
def load_streamlit_df():
    return pd.read_csv("demo_data/df_streamlit_subset.csv",index_col=0)

def randomInput():
    df = load_streamlit_df()
    # df = df[(df["prediction_correct_image"]==True) & (df["prediction_correct_text"]==True)]
    product = df.sample()
    st.session_state['designation_input'] = ''
    st.session_state['description_input'] = str(product.iloc[0]['text_fr'])
    st.session_state['class_input'] = prdcodetype2label[product.iloc[0]['prdtypecode']]
    print("image_file=",product.iloc[0]['imagefile'])
    st.session_state['image_input'] = product.iloc[0]['imagefile']
    st.session_state['scrap_input'] = ''

def clearForm():
    st.session_state['designation_input'] = ''
    st.session_state['description_input'] = ''
    st.session_state['scrap_input'] = ''
    st.session_state['class_input'] = ''
    st.session_state['image_input'] = ''
    st.session_state['image_url'] = ''


def renderDemonstration():
    # speed loading of app by loading models just now
    utils_camembert.init()
    utils_vgg16.init()
    st.title('Démonstration')
    st.divider()

    with st.container():
        col1,col2=st.columns([2, 5])
        with col1:
            c1, c2 = st.columns(2)
            with c1:
                st.button('Aléatoire', use_container_width = True, on_click = lambda: randomInput())
            with c2:
                st.button('Effacer', use_container_width = True, on_click = lambda: clearForm())
            text_weight = st.slider('Ajuster le poids du modèle texte', 0.0, 1.0, 0.5, 0.01, label_visibility='collapsed')
            st.write(f"Poids Camembert: {text_weight:.2f} - Poids VGG16: {1 - text_weight:.2f}")
            with st.form('predict_form', clear_on_submit=False):       
                designation = st.text_input(
                    "Désignation",
                    placeholder = "Désignation et description du produit",
                    value=st.session_state['designation_input'],
                    key='designation_input'
                )    
                description = st.text_area(
                    "Description",
                    placeholder = "Description du produit",
                    value = st.session_state['description_input'],
                    key='description_input'
                )

                rakuten_url = st.text_input(
                    "Rakuten URL", 
                    placeholder = 'Lien vers produit Rakuten',
                    value = st.session_state['scrap_input'],
                    key='scrap_input'
                )

                uploaded_file = st.file_uploader("Importer une image", type=['png', 'jpg'])
                if uploaded_file is not None:
                    uploaded_file = Image.open(uploaded_file)
                
                submitted = st.form_submit_button('Prédire', use_container_width=True)

        with col2:
            if submitted:
                with st.spinner('Classification en cours, veuillez patienter....'):
                    if rakuten_url:
                        try:
                            designation, description, image_url = scrap(rakuten_url)
                            print(image_url)
                            uploaded_file = Image.open(requests.get(image_url, stream=True).raw)
                        except:
                            designation, description, image_url = None, None, None
                            st.text('Impossible de charger l\'URL')
                    if designation or description:
                        text_predictions = utils_camembert.predict(designation + " " + description)
                        # print("text_prediction=",
                        #         prdcodetype2label[utils_camembert.classes_order[np.argmax(text_predictions)]] )
                    if uploaded_file is not None:
                        uploaded_file.load()
                        img_resized = uploaded_file.resize((224, 224))
                        img_array = np.asarray(img_resized)
                        image = img_array.reshape((224, 224, 3))
                        image_predictions = utils_vgg16.predict(image,utils_camembert.classes_order)
                        # print("image_prediction=",
                        #         prdcodetype2label[utils_camembert.classes_order[np.argmax(image_predictions)]] )
                    if (designation or description) and uploaded_file is not None:
                        predictions = get_average_pred(image_predictions, text_predictions, text_weight)
                    else:
                        try:
                            image_predictions
                        except NameError:
                            print('Image not defined')
                        else: 
                            predictions = image_predictions
                        try:
                            text_predictions
                        except NameError:
                            print('Text not defined')
                        else:
                            predictions = text_predictions[0]
                
                try: 
                    predictions
                except NameError:
                    st.text('No predictions found')
                else:
                    class_labels = [prdcodetype2label[class_label] \
                        for class_label in utils_camembert.classes_order ]
                    class_predictions = list(zip(class_labels, predictions))

                    # Trier les prédictions par probabilité (du plus élevé au plus bas)
                    sorted_predictions = sorted(class_predictions, key=lambda x: x[1], reverse=True)

                    fig, ax = plt.subplots(figsize=(15, 4))
                    ax.bar(*zip(*sorted_predictions[:10]), color='#ff4148')
                    ax.set_title('Prédictions par catégorie', fontsize=16)
                    ax.set_ylabel('Probabilités', fontsize=14)
                    plt.xticks(rotation=40)
                    st.pyplot(fig)

                    categorie, taux_de_confiance = sorted_predictions[0]

                    # Formatage du texte en Markdown
                    prediction_markdown = f"<h3 style='font-size:1.6em;'>Prédiction finale : <strong style='color:#ff4148;'>{categorie}</strong> </h3>"
                    class_markdown = f"<h3 style='font-size:1.6em;'>Classe réelle : <strong style='color:#ff4148;'>{st.session_state['class_input']}</strong> </h3>"
                    rate_markdown = f"<h3 style='font-size:1.6em;'>Taux de confiance : <strong style='color:#ff4148;'>{taux_de_confiance:.2%}</strong> </h3>"

                    # Affichage du texte en Markdown avec Streamlit
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(prediction_markdown, unsafe_allow_html=True)
                        if st.session_state['class_input'] != '':
                            st.markdown(class_markdown, unsafe_allow_html=True)
                        st.markdown(rate_markdown, unsafe_allow_html=True)
                    with col2:
                        if uploaded_file is not None:
                            st.image(uploaded_file, width=200)

            else:
                c1, c2, c3 = st.columns([1, 5, 1])
                with c2:
                    lottie = load_lottiefile('assets/dashboard.json')
                    st_lottie(lottie,key='demo', width=600)