import streamlit as st
from streamlit_option_menu import option_menu

def renderPreprocessing():
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
        st.header('Preprocessing des images')
        st.markdown("""
            - Déplacement de chaque image dans un sous-dossier propre à sa classe d'appartenance
            - Ajout d'une colonne imagefile au Dataframe contenant leur chemin relatif
            - Zoom des images centrales ayant un ratio inférieur ou égal à 80%
        """)
        st.image("assets/zoom_images.png")
