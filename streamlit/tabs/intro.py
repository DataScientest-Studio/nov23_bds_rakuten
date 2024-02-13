import streamlit as st
def renderIntroduction():
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