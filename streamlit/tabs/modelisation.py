import streamlit as st
from streamlit_option_menu import option_menu

def renderModelisation():
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