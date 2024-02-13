import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from prdcodetype2label import prdcodetype2label

def renderExploration(df):
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
                st.markdown("""
                        - En parcourant les données tabulaire, 
                        nous avons observé un grand nombre de textes contenant du html 
                        soit sous forme de tags, 
                        soit sous forme de caractères encodés.
                        - Dans le but de faciliter l’analyse des langues et de la fréquence des mots, 
                        nous avons créé une fonction permettant de supprimer le html et 
                        remplacer les caractères encodés que nous avons appliqué 
                        aux variables designation et description.
                """)
                st.image('assets/html_proportion.png')
        with st.expander('Analyse des langues'):        
            st.header("Analyse des langues")
            with st.container():
                st.markdown("""
                    - Lors de nos explorations de données, 
                    nous avons également remarqué la présence de texte dans plusieurs langues, 
                    principalement en français, anglais et allemand.
                    - Pour clarifier la situation, nous avons décidé d’utiliser 
                    la librairie langdetect pour détecter la langue la plus probable 
                    de chaque texte
                    - Les modèles NLP étant en général orienté vers une langue particulières, 
                    nous avons alors envisagé nos options:
                        - Supprimer les données dans une langue autre que le français
                        - Traduire toutes les langues vers l’anglais
                        - Traduire toutes les langues vers le français

                """)
                st.image("assets/lang_pie.png")
                st.image('assets/expl_text_1.png')

        
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


        with st.expander('Fréquence des mots par type de produit'):
            st.header('Fréquence des mots par type de produit')
            

    if selected2=='Images':
        st.header('Analyse des contours')
        st.image('assets/img_zoom.png')