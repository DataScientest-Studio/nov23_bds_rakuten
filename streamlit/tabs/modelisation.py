import streamlit as st
from streamlit_option_menu import option_menu

def renderModelisation():
    st.title('Modélisation')
    st.divider()

    option = option_menu(None, ["Texte", 'Images', 'Fusion'], 
        icons=['chat-text', "images", "file-richtext"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if option == 'Texte':
        st.header('Modèle de classification de texte')
        st.subheader("*Étude en 3 phases*")
        with st.expander('Phase 1 : Modèles Classiques'):
            st.header('Modèles Classiques')
            st.markdown("""
            Application de techniques statistiques et d'apprentissage machine traditionnel pour la 
            classification de texte, avec une préparation spécifique du texte.

            1. Préparation du Texte
                - Traduction
                - Filtrage stop Words
                - WordNetLemmatizer
                - TF-IDF
     
            2. Modèles testés : 
                - Random Forest
                - Gradient Boosting
                - SVM
            3. Performance 
                Accuracy: Varie de 0,3 à 0,81
            """)

            st.subheader("Conclusion")
            st.markdown("""
            Importance du préprocessing et des paramètres d'entrainement sur les résultats
            Meilleur score obtenu avec SVM et Filtrage stop words + WordNetLemmatizer + TF-IDF
            """)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader('Rapport de classification')
                st.image('assets/cf-svm-10.png', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/confusion-matrix-svm-10.png', width=600)
            
        with st.expander('Phase 2 : Réseau de Neurones'):
            st.header('Réseau de Neurones')
            st.markdown("""
            Utilisation de réseaux de neurones pour traiter des tâches de NLP,
            pour capturer des relations complexes dans les données textuelles.

            1. Préparation du Texte
                - Traduction
                - Filtrage stop Words
                - WordNetLemmatizer
                - CBOW
                - Skip Gram
                - Tokenisé + padding
     
            2. Modèles testés : 
                - RNN avec GRU
                - FastText (Hybride classique / RNN)

            3. Performance 
                - Accuracy: Autour de 0,7 à 0,74 pour RNN GRU
                - Accuracy: Varie de 0,47 à 0,8 pour FastText
            """)

            st.subheader("Conclusion")
            st.markdown("""
            Approche FastText extrèmement efficace en terme de rapidité et de résultat. Beaucoup de paramètres à maîtriser pour 
            optimiser
            """)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader('Rapport de classification')
                st.image('assets/cf-FastText-21.png', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/confusion-matrix-FastText-21.png', width=600)
             
        with st.expander('Phase 3 : Transfert Learning'):
            st.header('Transfert Learning')
            st.markdown("""
            Utilisation de modèles pré-entraînés adaptés à de nouvelles tâches, 
            permettant d'exploiter des connaissances linguistiques complexes acquises sur de vastes corpus.

            1. Préparation du Texte
                - Traduction
                - Tokeniser / encoder avec CamemBERT
     
            2. Modèles testés : 
                - CamemBERT

            3. Performance 
                - Accuracy: Jusqu'à 0,89
            """)

            st.subheader("Conclusion")
            st.markdown("""
            L'utilisation de transfert learning permet d'obtenir des résultats significativement meilleurs, 
            malgrès le peu d'ajustements sur les paramètres d'entraînement.
            """)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader('Rapport de classification')
                st.image('assets/cf-CamemBERT-28.png', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/confusion-matrix-CamemBERT-28.png', width=600)
       
    if option == 'Images':
        st.header("Modèles d'images")
        st.subheader("*Trois architectures CNN testées*")
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
            st.markdown("""
                Dans le but d'améliorer les performances de notre modèle, nous avons utilisé 
                la méthode de transfert learning avec, comme architecture de détection de caractéristiques,
                le modèle ResNet152 avec auquel nous avons rajouté plusieurs couches de classifications permettant
                ainsi de répondre à notre besoin initial. Nous avons testé différents hyper-paramètres en se basant sur
                les tests précédemment faits sur le modèle LeNet. Ensuite, nous avons tenté de réduire l'overfitting au maximum
                en essayant plusieurs couches de classifications différentes. Enfin, nous sommes arrivés à weighted-f1 score de 0.58
                avec une bonne généralisation malgré certaines lacunes dans quelques catégories.
            """)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader('Rapport de classification')
                st.image('assets/cf_resnet.png', width=400)
            with col2:
                st.subheader('Matrice de confusion')
                st.image('assets/heatmap_resnet.png', width=800)

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

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("""
                    #### Rapport de classification
                """)
                st.image("assets/classification-report-vgg16.png", width=400)
            
            with c2:
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
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("""
                #### Rapport de classification
            """)
            st.image("assets/classification-report-fusion.png", width=400)
        
        with c2:
            st.markdown("""
                #### Matrice de confusion
            """)
            st.image("assets/confusion-matrix-fusion.png")
        