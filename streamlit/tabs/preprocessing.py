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
        
        with st.expander('Techniques de préparation du texte'):
            #st.header('Préparation du texte')
            st.markdown("""
                **Préparation du texte**  
                Application de techniques statistiques et d'apprentissage machine traditionnel pour la classification de texte, avec une préparation spécifique du texte.

                **1. Nettoyage du Texte:**  
                Suppression des balises Html, de la ponctuation.

                **2. Tokenisation:**  
                Division du texte en mots ou phrases (tokens) pour faciliter leur traitement.

                **3. Suppression des mots vides (stop words):**  
                Les mots très fréquents (comme "le", "et", "à") qui n'apportent pas de valeur significative pour l'analyse sont retirés.

                **4. Normalisation:**  
                Application de la lemmatisation ou du stemming pour réduire les mots à une forme de base.

                **5. Vectorisation et padding:**  
                Transformation du texte en vecteurs numériques que le modèle peut traiter. Le sens des phrases peut être perdu dans cette étape.                                                        
                Ceci peut être réalisé par des techniques comme:
                - Word2Vec : CBOW (Continuous Bag of Words) ou Skip Gram
                - TF-IDF
                - Des embeddings contextuels (BERT, GPT)
                """, unsafe_allow_html=True)
            
        with st.expander('Techniques principales appliquées'):        
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
