# Rakuten France Multimodal Product Data Classification

## Presentation

This repository contains the code for our project **Rakuten product classification**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The project is issued from the Rakuten France Multimodal Product Data Classification challenge. Datas and their descriptions are available publicly here : https://challengedata.ens.fr/challenges/35

The goal of the project is to classify products based on a some text and an image describing the product.

This project was developed by the following team :

- Julien Noel du Payrat ([GitHub](https://github.com/surfncode) / [LinkedIn](https://www.linkedin.com/in/jundp/))
- Karim Hadjar ([LinkedIn](https://www.linkedin.com/in/karim-hadjar-52059b268/))
- Mathis Poignet ([LinkedIn](https://www.linkedin.com/in/mathispoignet/))

## Architecture

We have organized the git repository as follows:

- **data:** Contains the provided data CSVs. We didn't add the images to git, as the folder was too large. However, a zip file is publicly accessible on Google Drive: [images.zip](https://drive.google.com/file/d/1Qi_gEQet9Yls5vKGr5erRqspjpno3c0B/view?usp=drive_link).
- **notebooks:** Contains the notebooks to be executed in the listed order. Indeed, most notebooks produce results that other notebooks rely on.
    - **data-exploration:** Data exploration and visualization.
    - **data_preprocessing_traduction_fr:** French text translation.
    - **data_preprocessing_images:** Image zooming.
    - **data_preprocessing_resampling:** Class rebalancing.
    - **Data-preprocessing-text-stopWord-Steming:** Stop words filtering, text tokenization, and lemmatization.
    - **data-modeling-images-1:** Image modeling with LeNet.
    - **data-modeling-images-2:** Image modeling with ResNet152.
    - **data-modeling-images-3:** Image modeling with VGG16.
    - **data-modeling-text-1-TF-IDF:** Text modeling with TF-IDF.
    - **data-modeling-text-1bis-TF-IDF:** Continuation of text modeling with TF-IDF.
    - **data-modeling-text-2-Cbow:** Text modeling with Cbow.
    - **data-modeling-text-3-Skip Gram:** Text modeling with Skip Gram.
    - **data-modeling-text-4-RNN-GRU:** Text modeling with RNN GRU.
    - **data-modeling-text-5-Fasttext:** Text modeling with Fasttext.
    - **data-modeling-text-6-CamenBERT:** Text modeling with Camembert.
    - **data-modeling-text-6 retrain-CamenBERT:** Continuation of text modeling with Camembert.
    - **data-modeling-fusion:** Fusion of text and image models.
    - **data-modeling-interpretation-images**: Late addition of activation maximization technique to interpret results of VGG16 best model .
    - **streamlit-data-preparation**: Late addition of code to prepare a data subset for streamlit demo
- **output:** Contains the results of the notebooks, each in a subfolder named identically to the corresponding notebook.
- **assets:** Contains some image resources used by the notebooks.
- **reports:** Contains this report in PDF format.
- **streamlit:** Contains the streamlit demo

## Running the notebooks

We made some efforts to allow running the notebooks both locally and on google colab. However, due to a lack of proper hardware on our personal computers, we were not able to fully test the notebooks which needed a dedicated hardware (GPU).

### Running on google colab

1. Transfer the current git repository on a google drive. Make sure to keep its name identical (**nov23_bds_rakuten**)
2. The images were not stored on git due to space limitations. You must download them from our google drive via this link: [images.zip](https://drive.google.com/file/d/1Qi_gEQet9Yls5vKGr5erRqspjpno3c0B/view?usp=drive_link)
3. You need to copy **images.zip** into **nov23_bds_rakuten/data** on your google drive

You're all set up. You should now be able to open the notebooks from **nov23_bds_rakuten/notebooks** on google colab. If you plan to re-run them, please be mindful of the order indicated in the section **Architecture**

### Running locally


1. The images were not stored on git due to space limitations. You must download them from our google drive via this link: [images.zip](https://drive.google.com/file/d/1Qi_gEQet9Yls5vKGr5erRqspjpno3c0B/view?usp=drive_link)
2. You need to copy **images.zip** into **nov23_bds_rakuten/data** on your computer
3. Extract **images.zip** into a folder named **nov23_bds_rakuten/data/images** (please also leave **nov23_bds_rakuten/data/images.zip** as some notebooks depend on it)
4. Install the requirements (preferably in a python venv or conda environment)
```bash
pip install -r requirements.txt
```

You're all setup. You should now be able to open the notebooks. If you plan to re-run them, please be mindful of the order indicated in the section **Architecture**

## Running the streamlit demo

1. The models were too big to store on git. You must download them separately, and extract them into **streamlit/models**. Here is the public download link located on our google drive: [models.zip](https://drive.google.com/file/d/1mvkmmFL5efTv60xwsZkpRmnywUzmCCcR/view?usp=sharing).

2. Next, type the following commands in a terminal, preferably in a virtual environment like venv or conda (starting from root of git repo):
```bash
cd streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```
That's it, you should now have the url of streamlit app (http://localhost:8501/) opened in your default browser !


