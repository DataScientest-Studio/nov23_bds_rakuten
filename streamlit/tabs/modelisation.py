import streamlit as st
from streamlit_option_menu import option_menu

def renderModelisation():
    st.title('Modélisation')
    st.divider()

    option = option_menu(None, ["Texte", 'Images', 'Fusion'], 
        icons=['chat-text', "images", "file-richtext"], 
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
    if option == 'Images':
        st.header("Modèles d'images")
        st.markdown("Trois architectures CNN testées")
        with st.expander('Modèle d\'image LeNet'):
            st.header('Modèle d\'image LeNet')
            st.markdown("""
            CNN simple qui nous a permis de nous faire une première idée de la contribution 
            des différents facteurs (données, hyper-paramètres) à la performance de notre modèle.

            1. On a d'abord tenté de mesurer l'impact des différentes étapes de prétraitement sur nos modèles. 
            Toutes choses égales par ailleurs, on a testé le même modèle avec, dans l'ordre:
                1. Le dataset sans rééquilibrage de classes et avec les images non zoomées
                2. Le dataset sans rééquilibrage de classes et avec les images zoomées
                3. Le dataset avec rééquilibrage de classes et avec les images zoomées
            2. Enfin, après avoir trouvé le dataset le plus approprié, 
            on a testé l'impact de différents hyper-paramètres sur la performance:
                1. Un learning rate plus élevé avec une stratégie de learning rate decay
                2. Un batch_size plus gros
                3. Une taille d'image plus grande
            3. On a sélectionné le dataset et les hyper-paramètres les plus prometteurs des sections 
            précédentes et entraîné le modèle de façon plus poussée sur un dataset complet
            """)

            st.subheader("Conclusion")
            st.image("assets/recap-lenet.png")
            st.markdown("""
            Les performances obtenues avec le dernier modèle étaient encore loin des 0.5-0.6 
            de f1-score qu'on aurait aimé atteindre
            """)

        with st.expander('Modèle d\'image ResNet152'):
            st.header('Modèle d\'image ResNet152')
            st.subheader('Rapport de classification')
            st.image('assets/cf_resnet.png', width=400)
            st.subheader('Matrice de confusion')
            st.image('assets/heatmap_resnet.png')

        with st.expander('Modèle d\'image VGG16'):
            st.header('Modèle d\'image VGG16')
            st.markdown("""
            N'ayant pu atteindre un niveau de performance adéquat dans un temps raisonnable, 
            nous avons cette fois utilisé sur un modèle de transfert learning en utilisant une 
            architecture VGG16 avec les poids d'imagenet.

            Nous avons procédé en cinq étapes:

            1. Effectuer un entraînement de test sur une portion limitée du dataset pour 
            évaluer le choix de nos hyper-paramètres initiaux.
            2. En se basant sur l'évaluation du premier test, 
            lancer un entraînement sur les données complètes
            3. Ré-entraîner les 4 dernières couches de VGG16 pour tenter d'ajuster au plus près 
            les poids de la partie extraction de features du meilleur modèle de 1 et 2
            4. Sélectionner le modèle le plus prometteur puis refaire des entraînements en 
            tentant de résoudre les problèmes d'overfitting observés
            5. Ré-entraîner les 4 dernières couches de VGG16 pour tenter d'ajuster au plus près les 
            poids de la partie extraction de features du meilleur modèle de 4
            """)

            st.subheader("Conclusion")
            st.image("assets/recap-vgg16.png")
            st.markdown("""
            Nous avons obtenu les meilleurs résultats avec le modèle de l’étape 3 (id 331) 
            qui a atteint un f1-score de 0.60.
            """)

            st.markdown("""
                #### Rapport de classification
            """)
            st.image("assets/classification-report-vgg16.png")

            st.markdown("""
                #### Matrice de confusion
            """)
            st.image("assets/confusion-matrix-vgg16.png")
    if option == 'Fusion':
        st.header("Fusion des modèles de texte et d’image")
        st.markdown("""
        L'objectif du notebook data-modeling-fusion était de tester des modèles d'ensemble 
        qui permettent de fusionner les prédictions des meilleurs modèles que nous avions pu 
        trouver au cours de nos modélisations sur les textes et les images.
        
        Nous avons expérimenté deux grand types de modèles d'ensemble en trois étapes:

        1. Un modèle classique de fusion qui reprend les principes de la classe 
        sklearn.ensemble.VotingClassifier à laquelle on aurait passé le paramètre voting="soft".
        2. Une émulation de sklearn.ensemble.StackingClassifier utilisant LogisticRegression 
        comme classifieur final.
        3. Nous avons réutilisé le modèle de fusion de 1 avec un nouveau modèle texte basé 
        sur CamemBERT 
        """)

        st.subheader("Conclusion")
        st.image("assets/recap-fusion.png")
        st.markdown("""
        Nous avons obtenu les meilleurs résultats avec le modèle de l’étape 3 (id 431) 
        qui a atteint un f1-score de 0.8907.
        """)

        st.markdown("""
            #### Rapport de classification
        """)
        st.image("assets/classification-report-fusion.png")

        st.markdown("""
            #### Matrice de confusion
        """)
        st.image("assets/confusion-matrix-fusion.png")
        