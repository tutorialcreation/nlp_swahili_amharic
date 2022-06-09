# African Language Speech Recognition Modeling 
## Table of Contents
- [African Language Speech Recognition Modeling](#African-Language-Speech-Recognition-Modeling)
  - [Project Overview](#project-overview)
  - [Business Objective](#business-objective)
  - [Skills Implemented in the Project](#skills-implemented-in-the-project)
  - [ML Pipeline Design](#ML-pipeline-design)
  - [Manual Installation](#manual-installation)
  - [Automatic Installation](#automatic-installation)
  - [Modularized Script Tests](#modularized-script-tests)
  - [Dataset Column Description](#dataset-column-description)

## Project Overview
Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond to more and more of our verbalized commands. There are many matured speech recognition systems available, such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for limited languages only. 


## Business Objective
The World Food Program intends to use an intelligent form to collect nutritional information about food purchased and sold at markets in two African countries: Ethiopia and Kenya. The intelligent form's design requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app, which registers the list of items they just bought in their own language. The app's intelligent systems are expected to live to transcribe speech-to-text and organize information in an easy-to-process database.
We will provide speech-to-text technology in two languages: Amharic and Swahili. It is our responsibility to create a deep learning model capable of transcribing a speech to text. The model should be accurate and resistant to background noises. 

## Skills Implemented in the Project
* Working with audio as well as text files
* Familiarity with the deep learning architectures
* Model management (building ML catalog containing models, feature labels, and training model version)
* Comparing multiple Deep learning techniques; 
* Training and validating DL models; 
* Choosing appropriate architecture, loss function, and regularisers; hyperparameter tuning; choosing suitable evaluation metrics. 
* MLOps  with DVC, CML, and MLFlow

## ML Pipeline Design
![Test Image 4](https://miro.medium.com/max/1400/1*rBUXN2u1Yh-9pxKzUGjmMg.png)

## Manual Installation
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
### Step 4: Visualize ML pipeline
```
dvc dag
```

## Automatic Installation
### Step 1: Docker
```
docker-compose up --build
```

## Modularized Script Tests
The tests from the modularized scripts are available in the following notebooks
* EDA analysis and Preprocessing ==> notebooks/Audio_preprocessing.ipynb
* Deep learning ==> notebooks/Audio_Modelling.ipynb

## Dataset Column Description
* Input features (X): audio clips of spoken words
* Target labels (y): a text transcript of what was spoken
