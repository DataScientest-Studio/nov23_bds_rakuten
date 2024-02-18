import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from tabs import intro, exploration, preprocessing, modelisation, demonstration, conclusion
from prdcodetype2label import prdcodetype2label
from utils import load_lottiefile, pull_clean

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
[data-testid='stAppViewBlockContainer']{
            padding: 3rem 5rem 2rem 5rem;
}
</style>
""", unsafe_allow_html=True)

df = pull_clean()

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
        padding: 4rem 1.5rem 2rem 1.5rem;
    }
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    selected = option_menu('Projet Rakuten', ["Introduction", "Exploration", "Preprocessing", "Modélisation", "Démonstration", "Conclusion"], 
        icons=['play-btn','bar-chart','gear', 'diagram-3', 'play', 'activity'],menu_icon='collection-play', default_index=0, key='main')
    
    lottie = load_lottiefile('assets/process.json')
    st_lottie(lottie,key='sidebar', width=250)

# Intro
if selected=="Introduction":
    intro.renderIntroduction()

# Exploration
if selected=="Exploration":
   exploration.renderExploration(df)

# Preprocessing
if selected=='Preprocessing':
    preprocessing.renderPreprocessing()

# Modélisation
if selected=='Modélisation':
    modelisation.renderModelisation()

if selected=='Démonstration':
    demonstration.renderDemonstration()

# Conclusion
if selected=="Conclusion":
    conclusion.renderConclusion()



