import streamlit as st

def renderAbout():
    st.title('Présentation des membres du projet')
    st.subheader('_Promotion Bootcamp Novembre 2023_')
    
    st.divider()

    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            st.image('assets/profile_pic_1.png', width=250)
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
            st.image('assets/profile_pic_2.png', width=250)
            st.header('Karim Hadjar')
            st.markdown(
                """
                - Expérience dans la création de tableaux de bord et l'utilisation d'outils (ex: Excel, Power BI)
                - Approche empirique privilégiée pour l'exploration des données et la sélection des visualisations..

                - [Linkedin](https://www.linkedin.com/in/karim-hadjar-52059b268/)
                """
                )
        with col3:
            st.image('assets/profile_pic_3.png', width=250)
            st.header('Mathis Poignet')
            st.markdown(
                """
                - Sorti d'école d'ingénieur, je me spécialise dans le domaine de la data science
                - Première expérience en data science lors de mon stage d'IUT (segmentation d'image)
                - [Linkedin](https://www.linkedin.com/in/mathispoignet/)
                """
                )