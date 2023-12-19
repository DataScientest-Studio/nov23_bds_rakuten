# Rakuten France Multimodal Product Data Classification

## Presentation and Installation

This repository contains the code for our project **Rakuten product classification**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The project is issued from the Rakuten France Multimodal Product Data Classification challenge. Datas and their descriptions are available publicly here : https://challengedata.ens.fr/challenges/35

The goal of the project is to classify products based on a some text and an image describing the product.

This project was developed by the following team :

- Julien Noel du Payrat ([GitHub](https://github.com/surfncode) / [LinkedIn](https://www.linkedin.com/in/julien-noel-du-payrat-01854558))
- Karim Hadjar
- Mathis Poignet

You can browse and run the [notebooks](./notebooks). 
The notebooks are meant to be run on [google colab](https://colab.research.google.com/). You need a google drive hosting the data. Please download the following zip, and extract it in a folder named **Projet_Rakuten** at the root of your drive: (TODO: link to zip containing the data)

TODO: see how to give the choice of running the notebooks locally.

TODO: complete the next sections

You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
