import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from prdcodetype2label import prdcodetype2label

def renderExploration(df):
    st.title('Exploration du jeu de données')
    st.divider()

    selected2 = option_menu(None, ["Contexte", "Texte", 'Images'], 
        icons=['database', 'chat-text', "images"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if selected2=='Contexte':
        with st.expander('Challenge Rakuten'):
            st.header("Challenge Rakuten")
            st.markdown("""
            Ce projet s’inscrit dans le challenge Rakuten France Multimodal Product Data Classification, 
            les données et leur description 
            sont disponibles à l’adresse : https://challengedata.ens.fr/challenges/35

            - _Données textuelles : ~60 mb_
            - _Données images : ~2.2 gb_
            - _99k données réparties en 27 classes._
            - _Métrique cible: weighted-f1 score_
            """)

        with st.expander('Fichiers fournis'):
            st.header("Fichiers fournis")
            st.markdown("""
            - **X_train.csv**: Contient les variables explicatives destinées à l’entraînement des modèles.
                - **index** (nb entier): Index du produit.
                - **designation** (object: string): Designation courte du produit
                - **description** (object: string, optionnel): Description du produit. 
                Ce champ est optionnel. Tous les produits n'ont pas de description
                - **productid** (int64): L'id du produit
                - **imageid** (int64): L'id de l'image du produit
            - **images.zip**: Une fois extrait, un dossier contenant les images des produits. 
            La nomenclature utilisée permet de faire la jonction avec les produits. 
            Chaque fichier d'image se présente sous la forme: image_<imageid>_product_<productid>.jpg. 
            Les images sont répartis en deux sous-dossiers:
                - **image_train**: Les images correspondants à **X_train.csv**
                - **image_test**: Les images correspondants à X_test.csv (pas utilisées)
            - **Y_train.csv**: Contient la variable cible à prédire. A savoir le type du produit.
                - **index** (nb entier): Index du produit. Permet de faire la jonction avec X_train.csv
                - **prdtypecode** (nb entier): Le type du produit
            """)

        with st.expander('Aperçu initial'):
            st.header("Aperçu initial")
            st.markdown("""
            Nous avons chargé le fichier **X_train.csv** et effectué une première inspection des données.
            """)
            st.dataframe(df[['designation', 'description', 'productid', 'imageid']].head().style.format(thousands=''), hide_index=True)
            st.markdown("""
            - Le fichiers contenait 84916 observations
            - Le champ **description** contenait un grand nombre de valeurs nulles (35%), 
            ce qui était attendu pour un champ optionnel.
            - Les autres colonnes ne présentaient pas de valeurs nulles.
            """)

            # Analyse variable cible
            st.subheader("Analyse de la variable cible")

            st.markdown("""
            Nous avons également chargé le fichier **Y_train.csv** et constaté qu'il n'y avait pas 
            de valeurs manquantes. 
            Les deux fichiers avaient le même nombre d'entrées, 
            ce qui indiquait une correspondance totale entre les deux. 
            Correspondance qui s’est vérifiée lors de la fusion des variables explicatives et la variable cible 
            dans un DataFrame unique.
            """)

            df['categorie'] = df['prdtypecode'].map(prdcodetype2label)
            fig, ax = plt.subplots(figsize=(15, 3))
            sns.set(style="whitegrid")
            sns.countplot(x=df['prdtypecode'], order=df['prdtypecode'].value_counts().index, color='#ff4148')
            plt.title('Nombre de produits par prdtypecode', fontsize=14)
            plt.xlabel('prdtypecode', fontsize=10)
            plt.ylabel('Nombre de produits', fontsize=10)
            st.pyplot(fig)
            st.markdown("""
                - 27 classes possibles (labellisées avec un dictionnaire)
                - Disparités significatives: Certaines classes sous-représentées (60: Console de jeu), 
                tandis que d'autres sur-représentées (2583: Autour de la piscine)
            """)

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

            st.markdown("""
                Proportion de valeurs manquantes dans le champ description: 35%

                Plusieurs options pour les traiter:

                1. Supprimer les lignes les contenant
                2. Les remplacer par des chaînes de caractères vides ?
                3. Concaténer les variables designation et description dans une troisième variable
            """)


        with st.expander('Fréquence des mots par type de produit'):
            st.header('Fréquence des mots par type de produit')

            st.markdown('Dix mots les plus fréquents dans chaque catégorie')
            st.image("assets/top-10-words.png")
            st.markdown("""
                - Dans la plupart des catégories des mots reflétant leur thème
                - Certains mots génériques tels que "peut", "plus", "être"
                - D'autres termes typiques d'un vocabulaire produit, tels que 
                "dimension", "taille", "longueur"
                - Des mots classiques du marketing de vente comme "haute", "qualité" et "facile"
            """)
            
            st.markdown('Nuages de mots pour chaque catégorie')
            st.image("assets/world-clouds.png")
            st.markdown("""
                - Certains mots plus en arrière-plan qui n'apparaissaient pas dans le top 10
            """)

    if selected2=='Images':
        st.markdown("""
            - Beaucoup d'images contiennent une petite image centrale entourée de beaucoup de blanc
            - Calcul de la bounding box de l'image centrale avec une fonction maison
            - 20% des images centrales avaient un ratio inférieur à 0.8
            - Elles pourraient bénéficier d'un zoom
        """)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image('assets/img_zoom.png')

