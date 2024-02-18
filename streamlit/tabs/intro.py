import streamlit as st
from streamlit_option_menu import option_menu

def renderIntroduction():
    #Header
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.title('Présentation du projet')
        st.subheader('*Rakuten France Multimodal Product Data Classification*')
    with c2:
        st.image('assets/rakuten.png', width=200)
    with c3:
        st.image('assets/mines.png', width=210)
    with c4:
        st.image('assets/datascientest.png', width=100)
    st.divider()        
    
    option = option_menu(None, ["Membres du projet", 'Objectif'], 
        icons=['people', "rocket-takeoff"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if option == 'Objectif':
        st.header('Objectif')
        st.markdown("""
        L'objectif du projet est de cataloguer des produits selon un code type désignant le produit.
        La prédiction du type doit se faire à partir de données textuelles 
        (désignation et description du produit) ainsi que de données visuelles (image du produit).

        Étant réalisé dans le cadre de la formation Datascientest, 
        c'était l'opportunité pour nous de découvrir et mettre en application 
        des techniques de machine learning avancées telles que:

        - Computer vision
        - Réseaux de neurones convolutifs
        - NLP
        - Modèles multimodaux
        - Deep learning
        """)
    
    if option == 'Membres du projet':
        st.header('Membres du projet')
        st.subheader('_Promotion Bootcamp Novembre 2023_')

        with st.container():
            col1,col2,col3=st.columns(3)
            with col1:
                st.image('assets/profile_pic_1.png', width=225)
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
                st.image('assets/profile_pic_2.png', width=225)
                st.header('Karim Hadjar')
                st.markdown(
                    """
                    - Expérience dans la création de tableaux de bord et l'utilisation d'outils (ex: Excel, Power BI, ETL)
                    - Jusqu'ici, approche empirique pour l'exploration des données et la sélection des visualisations.
                    - [Linkedin](https://www.linkedin.com/in/karim-hadjar-52059b268/)
                    """
                    )
            with col3:
                st.image('assets/profile_pic_3.png', width=225)
                st.header('Mathis Poignet')
                st.markdown(
                    """
                    - Sorti d'école d'ingénieur, je me spécialise dans le domaine de la data science
                    - Première expérience en data science lors de mon stage d'IUT (segmentation d'image)
                    - [Linkedin](https://www.linkedin.com/in/mathispoignet/)
                    """
                    )