import streamlit as st

def renderConclusion():
	st.title('Conclusion')
	st.divider()
	
	st.header('Difficultés rencontrés lors du projet')
	st.markdown("""
	- Le preprocessing s'est révélé essentiel. 
	Nous avons réalisé l'importance de respecter l'ordre des étapes pour comprendre 
	les problèmes de performance.

	- Nous avons constaté la nécessité d'un matériel puissant pour le deep learning. 
	Même avec un accès payant à Google Colab, l'entraînement était lent.

	- Nous avons dû travailler sur un échantillon limité de données pour 
	réaliser des tests.

	- On a du rendre les notebooks résilients aux interruptions pour reprendre 
	le travail là où il s'était arrêté.

	- Nous avons compris l'importance d'un système de log pour gérer 
	 la multiplicité des tests de modélisation.
	""")

	st.header('Bilan')
	st.markdown("""
	L'utilisation de modèles pré-entraînés et le transfert learning peuvent 
	réduire considérablement le besoin en calcul en exploitant des caractéristiques 
	apprises sur de grands ensembles de données, 
	réduisant ainsi le temps nécessaire pour l'entraînement 
	sur un nouveau jeu de données.

	Nous sommes heureux d'avoir pu atteindre un niveau de performance en ligne avec 
	ce qu'on espérait (0.90 de f1-score)
	""")

	st.header('Suite du projet')
	st.markdown("""
	Pour surmonter les limitations des modèles transformers traditionnels, 
	l'exploration de modèles tels que BigBird, 
	conçus pour traiter efficacement de longs documents, 
	représente une voie prometteuse. 
	Ces modèles offrent une alternative pour gérer des textes excédant 
	largement la limite de 512 tokens, 
	potentiellement améliorant la compréhension contextuelle et la performance 
	globale sur des tâches de classification de textes complexes.

	Côté image, nous aurions aimé avoir le temps  d’expérimenter 
	des architectures CNN différentes telles que EfficientNet, 
	qui nous aurait sans doute permis d’augmenter la taille des images 
	fournies au réseau de neurones, 
	ou encore Inception pour sa capacité à faire le focus 
	sur les portions significatives des images.
	""")


