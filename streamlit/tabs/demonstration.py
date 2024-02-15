import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from PIL import Image
from prdcodetype2label import prdcodetype2label
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie 
from utils import load_lottiefile, get_average_pred
from utils_camembert import predictCamembert

#States
if 'designation_input' not in st.session_state:
    st.session_state['designation_input'] = ''
if 'description_input' not in st.session_state:
    st.session_state['description_input'] = ''
if 'image_input' not in st.session_state:
    st.session_state['image_input'] = ''
if 'scrap_input' not in st.session_state:
    st.session_state['scrap_input'] = ''

def randomInput(df):
    product = df.sample()
    st.session_state['designation_input'] = product.iloc[0]['designation']
    st.session_state['description_input'] = product.iloc[0]['description']
    st.session_state['image_input'] = 'image_' + str(product.iloc[0]['imageid']) + '_product_' + str(product.iloc[0]['productid']) + '.jpg'

def clearForm():
    st.session_state['designation_input'] = ''
    st.session_state['description_input'] = ''
    st.session_state['image_input'] = ''
    st.session_state['scrap_input'] = ''

def renderDemonstration(df, models):
    st.title('Démonstration')
    st.divider()
    
    with st.container():
        col1,col2=st.columns([2, 5])
        with col1:
            c1, c2 = st.columns(2)
            with c1:
                st.button('Aléatoire', use_container_width = True, on_click = lambda: randomInput(df[:10]))
            with c2:
                st.button('Effacer', use_container_width = True, on_click = lambda: clearForm())
            with st.form('predict_form', clear_on_submit=False):
                print(st.session_state['designation_input'])                
                designation = st.text_input(
                    "Désignation",
                    placeholder = "Désignation et description du produit",
                    value=st.session_state['designation_input']
                )
                description = st.text_area(
                    "Description",
                    placeholder = "Description du produit",
                    value = st.session_state['description_input']
                )

                # rakuten_url = st.text_input("Rakuten URL", placeholder = 'Lien vers produit Rakuten')

                uploaded_file = st.file_uploader("Importer une image", type=['png', 'jpg'])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    image.load()
                    image = image.resize((224, 224))
                    img_array = np.asarray(image)
                    uploaded_file = img_array.reshape((1, 224, 224, 3))
                
                submitted = st.form_submit_button('Prédire', use_container_width=True)

        with col2:
            if submitted:
                with st.spinner('Classification en cours, veuillez patienter....'):
                    if designation or description:
                        _, text_predictions = predictCamembert(designation + " " + description)
                    if uploaded_file is not None:
                        image_predictions = np.array(models['vgg16'].predict(uploaded_file)[0])
                    if (designation or description) and uploaded_file is not None:
                        predictions = get_average_pred(image_predictions, text_predictions)
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
                    class_predictions = list(zip(prdcodetype2label.values(), predictions))

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
                    rate_markdown = f"<h3 style='font-size:1.6em;'>Taux de confiance : <strong style='color:#ff4148;'>{taux_de_confiance:.2%}</strong> </h3>"

                    # Affichage du texte en Markdown avec Streamlit
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(prediction_markdown, unsafe_allow_html=True)
                        st.markdown(rate_markdown, unsafe_allow_html=True)
                    with col2:
                        if uploaded_file is not None:
                            st.image(uploaded_file)

            else:
                c1, c2, c3 = st.columns([1, 5, 1])
                with c2:
                    lottie = load_lottiefile('assets/dashboard.json')
                    st_lottie(lottie,key='demo', width=600)