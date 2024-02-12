import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import keras
from keras.models import load_model
import tensorflow as tf
from PIL import Image

#Layout
st.set_page_config(
    page_title="Projet Rakuten Challenge",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

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

models = load_models()
df = pull_clean()

prdcodetype2label = {
    10 : "Livre occasion",
    40 : "Jeu vidéo, accessoire tech.",
    50 : "Accessoire Console",
    60 : "Console de jeu",
    1140 : "Figurine",
    1160 : "Carte Collection",
    1180 : "Jeu Plateau",
    1280 : "Jouet enfant, déguisement",
    1281 : "Jeu de société",
    1300 : "Jouet tech",
    1301 : "Paire de chaussettes",
    1302 : "Jeu extérieur, vêtement",
    1320 : "Autour du bébé",
    1560 : "Mobilier intérieur",
    1920 : "Chambre",
    1940 : "Cuisine",
    2060 : "Décoration intérieure",
    2220 : "Animal",
    2280 : "Revues et journaux",
    2403 : "Magazines, livres et BDs",
    2462 : "Jeu occasion",
    2522 : "Bureautique et papeterie",
    2582 : "Mobilier extérieur",
    2583 : "Autour de la piscine",
    2585 : "Bricolage",
    2705 : "Livre neuf",
    2905 : "Jeu PC",
}

#Options Menu
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('assets/rakuten.png', width=140)
    hide_img_fs = '''
    <style>
    /*button[title="View fullscreen"]{
        visibility: hidden;}*/
    .st-emotion-cache-eczf16 {display: none}
    [data-testid='stSidebarUserContent']{
        padding: 4rem 1.5rem 2rem 1.5rem
    }
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    selected = option_menu('Projet Rakuten', ["Introduction", "Exploration", "Preprocessing", "Modélisation", "Démonstration", "Conclusion", "À propos"], 
        icons=['play-btn','bar-chart','gear', 'diagram-3', 'play', 'activity', 'info-circle'],menu_icon='collection-play', default_index=0, key='main')
    
    lottie = load_lottiefile('assets/process.json')
    st_lottie(lottie,key='sidebar', width=250)

# Intro
if selected=="Introduction":
    #Header
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        st.title('Présentation du projet')
        st.subheader('*Rakuten France Multimodal Product Data Classification*')
    with c2:
        st.image('assets/mines.png', width=400)
    with c3:
        st.image('assets/datascientest.png', width=160)

    st.divider()

    # Objectig
    st.header('Objectif')
    st.markdown("""<div style="text-align: justify;">L'objectif du projet est de cataloguer des produits selon un code type désignant le produit.
                La prédiction du type doit se faire à partir de données textuelles (désignation et description du produit) ainsi que de données 
                visuelles (image du produit).</div>""", unsafe_allow_html=True)

    st.divider()

    # Contexte
    st.header('Contexte')
    st.markdown(
        """
        Ce projet s’inscrit dans le challenge Rakuten France Multimodal Product Data Classification, les données et leur description 
        sont disponibles à l’adresse : https://challengedata.ens.fr/challenges/35
        - _Données textuelles : ~60 mb_
        - _Données images : ~2.2 gb_
        - _99k données réparties en 27 classes._
        """
        )

# Exploration
if selected=="Exploration":

    st.title('Exploration du jeu de données')
    st.divider()

    selected2 = option_menu(None, ["Données brutes", "Texte", 'Images'], 
        icons=['database', 'chat-text', "images"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if selected2=='Données brutes':
        st.header("Présentation des données brutes")
        st.dataframe(df[['designation', 'description', 'productid', 'imageid', 'prdtypecode']].head().style.format(thousands=''), hide_index=True)

        # Analyse variable cible
        st.header("Analyse de la variable cible")

        df['categorie'] = df['prdtypecode'].map(prdcodetype2label)
        fig, ax = plt.subplots(figsize=(15, 3))
        sns.set(style="whitegrid")
        sns.countplot(x=df['prdtypecode'], order=df['prdtypecode'].value_counts().index, color='#ff4148')
        plt.title('Nombre de produits par prdtypecode', fontsize=14)
        plt.xlabel('prdtypecode', fontsize=10)
        plt.ylabel('Nombre de produits', fontsize=10)
        st.pyplot(fig)

    if selected2=='Texte':
        with st.expander('Présence de HTML dans le texte'):
            st.header('Présence de HTML dans le texte')
            with st.container():
                col1,col2=st.columns(2)
                with col1:
                    st.image('assets/html_proportion.png')
                with col2:
                    st.markdown("""<div style="text-align: justify;">Beaucoup de textes dans les données ont du HTML, que ce soit avec des
                                balises comme <b></b> ou des caractères encodés comme &#FA;. On a remarqué
                                que les descriptions ont plus de HTML que les désignations. Pour simplifier l'
                                analyse des langues et des mots, on a fait une fonction pour enlever le HTML et 
                                remplacer les caractères encodés, utilisée sur "designation" et "description".</div>""", unsafe_allow_html=True)
                st.write(
                    """<style>
                    [data-testid="stHorizontalBlock"] {
                        align-items: center;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        with st.expander('Analyse des langues'):        
            st.header("Analyse des langues")
            with st.container():
                col1,col2=st.columns([1,2])
                with col1:
                    lang_simple = df['lang']
                    other_langs = df['lang'].value_counts().index[3:]
                    lang_simple = lang_simple.replace(other_langs, "other")
                    lang_counts = lang_simple.value_counts()

                    # Create a pie chart with Matplotlib
                    fig, ax = plt.subplots()
                    ax.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

                    # Set aspect ratio to be equal
                    ax.axis('equal')

                    # Display the pie chart
                    plt.title('Distribution des langues')
                    st.pyplot(fig)
                with col2:
                    st.text('te')
        
        with st.expander('Détection des valeurs manquantes'):
            st.header("Détection des valeurs manquantes")
            # Transforme la colonne categorie en string
            df['categorie'] = df['categorie'].astype(str)

            # Calcul du nombre d'apparitions de chaque catégorie
            category_counts = df['categorie'].value_counts()

            # Calcul des pourcentages de NaN et non-NaN pour chaque catégorie
            category_nan_counts = df.groupby('categorie')['description'].apply(lambda x: x.isna().sum())
            category_non_nan_counts = df.groupby('categorie')['description'].apply(lambda x: x.notna().sum())

            # Tri des catégories par nombre d'apparitions
            sorted_categories = category_counts.index.tolist()

            # Tri des pourcentages de NaN et non-NaN selon l'ordre des catégories
            sorted_nan_counts = category_nan_counts.reindex(sorted_categories)
            sorted_non_nan_counts = category_non_nan_counts.reindex(sorted_categories)

            # Création du graphique en barres
            fig, ax = plt.subplots(figsize=(15, 6))

            # Ajout des barres pour les valeurs non-NaN
            ax.bar(sorted_categories, sorted_non_nan_counts, label='Non-NaN', color='mediumseagreen')

            # Ajout des barres pour les valeurs NaN
            ax.bar(sorted_categories, sorted_nan_counts, label='NaN', bottom=sorted_non_nan_counts, color='#ff4148')

            # Paramètres du graphique
            plt.title('Répartition des NaN dans la description par catégorie', fontsize=14)
            plt.xlabel('Catégories', fontsize=10)
            plt.ylabel('Nombre de produits', fontsize=10)
            plt.xticks(rotation=60, ha='right')
            plt.legend()
            plt.tight_layout()

            # Affichage du graphique
            st.pyplot(fig)

        with st.expander('Répartition des langues par type de produit'):
            st.image('assets/expl_text_1.png')
        with st.expander('Fréquence des mots par type de produit'):
            st.header('Fréquence des mots par type de produit')
            

    if selected2=='Images':
        st.header('Analyse des contours')
        st.image('assets/img_zoom.png')

# Preprocessing
if selected=='Preprocessing':
    st.title('Démarche de préprocessing')
    st.divider()

    selected3 = option_menu(None, ["Texte", 'Images'], 
        icons=['chat-text', "images"], 
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected3=="Texte":
        st.header('Préparation du texte')
        row1 = st.columns(2)
        row2 = st.columns(2)

        texte = ["""
        **Texte non traduit avec Filtrage des stop words, WordNetLemmatizer, et TF-IDF :** 
        Cette approche a permis de réduire le bruit dans les données textuelles et de concentrer l'analyse sur les termes significatifs.""",

        """**Texte traduit avec CBOW (Continuous Bag of Words) :** 
        Les textes ont été vectorisés en utilisant l'approche CBOW de Word2Vec, permettant de capturer le contexte des mots dans les vecteurs.""",

        """**Skip Gram :** 
        Une autre technique de Word2Vec, Skip Gram, a été utilisée pour prédire le contexte à partir des mots, offrant une alternative à CBOW.""",

        """**Texte traduit tokenisé avec padding :** 
        Cette méthode a préparé les données pour les modèles de réseaux de neurones récurrents (RNN) en normalisant la longueur des séquences de texte.
        """]

        for i, col in enumerate(row1 + row2):
            tile = col.container(height=120)
            tile.markdown(texte[i])

    if selected3=='Images':
        st.header('Préparation des images')
        row1 = st.columns(3)
        row2 = st.columns(2)

        texte = ["""Merge des colonnes productid et imageid pour avoir les noms de fichiers""",
            """Répartition dans des sous dossiers selon leurs classes""",
            """Zoom sur les images""",
            """Équilibrage des classes""",
            """Export des données preprossed"""]
        
        for i, col in enumerate(row1 + row2):
            tile = col.container(height=120)
            tile.markdown(texte[i])

# Modélisation
if selected=='Modélisation':
    st.title('Modélisation')
    st.divider()

    option = option_menu(None, ["Texte", 'Image (ResNet152)', 'Image (VGG16)', 'Fusion'], 
        icons=['chat-text', "images", "images", "file-richtext"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if option == 'Texte':
        st.header('Modèle de texte CamenBERT')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader('Rapport de classification')
            st.image('assets/cf_resnet.png', width=400)
        with col2:
            st.subheader('Matrice de confusion')
            st.image('assets/heatmap_resnet.png')
    if option == 'Image (ResNet152)':
        st.header('Modèle d\'image ResNet152')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader('Rapport de classification')
            st.image('assets/cf_resnet.png', width=400)
        with col2:
            st.subheader('Matrice de confusion')
            st.image('assets/heatmap_resnet.png')
    if option == 'Image (VGG16)':
        st.header('Modèle d\'image VGG16')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader('Rapport de classification')
            st.image('assets/cf_vgg16.png', width=400)
        with col2:
            st.subheader('Matrice de confusion')
            st.image('assets/heatmap_resnet.png')
    if option == 'Fusion':
        st.header('Modèle de fusion Voting Classifier')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader('Rapport de classification')
            st.image('assets/cf_resnet.png', width=400)
        with col2:
            st.subheader('Matrice de confusion')
            st.image('assets/heatmap_resnet.png')
if selected=='Démonstration':
    st.title('Démonstration')
    st.divider()
    with st.container():
        col1,col2=st.columns([2, 5])
        with col1:
            option = st.selectbox(
                    'Choisir quel modèle utiliser : ',
                    ('Texte', 'Images', 'Fusion'),
                    label_visibility='collapsed'
                )
            
            with st.form('predict_form', clear_on_submit=True):

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


# Conclusion
if selected=="Conclusion":
    st.title('Conclusion')
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Bilan du projet')
    with col2:
        st.header('Difficultées rencontrées')
    with col3:
        st.header('Axes d\'amélioration')

# A propos
if selected=="À propos":
    st.title('Présentation des membres du projet')
    st.subheader('_Promotion Bootcamp Novembre 2023_')
    
    st.divider()

    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            st.image('assets/profile_pic_1.png', width=250)
            st.header('Julien Noel du Payrat')
            st.markdown(
                """
                - Background de développeur depuis plus de 15 ans
                - Première expérience en data science
                - [Linkedin](https://www.linkedin.com/in/julien-noel-du-payrat-01854558/)
                - [Github](https://github.com/surfncode)
                """
                )
        with col2:
            st.image('assets/profile_pic_2.png', width=250)
            st.header('Karim Hadjar')
            st.markdown(
                """
                - Expérience dans la création de tableaux de bord et l'utilisation d'outils (ex: Excel, Power BI)
                - Approche empirique privilégiée pour l'exploration des données et la sélection des visualisations..

                - [Linkedin](https://www.linkedin.com/in/karim-hadjar-52059b268/)
                """
                )
        with col3:
            st.image('assets/profile_pic_3.png', width=250)
            st.header('Mathis Poignet')
            st.markdown(
                """
                - Sorti d'école d'ingénieur, je me spécialise dans le domaine de la data science
                - Première expérience en data science lors de mon stage d'IUT (segmentation d'image)
                - [Linkedin](https://www.linkedin.com/in/mathispoignet/)
                """
                )


