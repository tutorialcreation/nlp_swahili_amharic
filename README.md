# Project Overview
Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond to more and more of our verbalized commands. There are many matured speech recognition systems available, such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for limited languages only. 


# Business Objective
The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. 
We shall be delivering a speech-to-text technology for two languages: Amharic and Swahili. Our responsibility is to build a deep learning model that is capable of transcribing a speech to text. The model produced should be accurate and robust against background noise.
 

# Skills implemented in the project:
* Working with audio as well as text files
* Familiarity with the deep learning architecture
* Model management (building ML catalog containing models, feature labels, and training model version)
* Comparing multiple Deep learning techniques; 
* Training and validating DL models; 
* Choosing appropriate architecture, loss function, and regularisers; hyperparameter tuning; choosing suitable evaluation metrics. 
* MLOps  with DVC, CML, and MLFlow

# ML Pipeline Design
![Test Image 4](https://miro.medium.com/max/1400/1*rBUXN2u1Yh-9pxKzUGjmMg.png)

# Manual Installation
### Step 1: Downloading source code
```
git clone https://github.com/tutorialcreation/nlp_swahili_amharic.git
```
### Step 2: Installation of dependencies
```
pip install -r requirements.txt
```
### Step 3: Check notebook
```
jupyter notebook
```
### Step 4: Visualize ML Pipeline
```
dvc dag
```

# Automatic Installation
### Step 1: Docker
```
docker-compose up --build
```

# The tests from the modularized scripts are run in the following notebooks
* EDA analysis and Preprocessing ==> notebooks/Audio_preprocessing.ipynb
* Deep learning ==> notebooks/Audio_Modelling.ipynb

# Dataset Column description
* Input features (X): audio clips of spoken words
* Target labels (y): a text transcript of what was spoken
