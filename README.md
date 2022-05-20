# Business Objective
The main objective of this project is to test if the ads that the advertising company runs resulted in a significant lift in brand awareness, click here for my medium blog on the topic.

# Project Overview
SmartAd is a mobile first advertiser agency. It designs Intuitive touch-enabled advertising. It provides brands with an automated advertising experience via machine learning and creative excellence. Their company is based on the principle of voluntary participation which is proven to increase brand engagement and memorability 10 x more than static alternatives. SmartAd provides an additional service called Brand Impact Optimiser (BIO), a lightweight questionnaire, served with every campaign to determine the impact of the creative, the ad they design, on various upper funnel metrics, including memorability and brand sentiment As a data scientist in SmartAd, in this project i will be designing a reliable hypothesis testing algorithm for the BIO service and determine whether a recent advertising campaign resulted in a significant lift in brand awareness.

# Skills implemented in the project:
* Statistical and Machine Learning Modelling
* Data science python libraries pandas, matplotlib, seaborn, scikit-learn
* Docker, DVC, Mlflow
* Linear regression
* Decision Trees
* XGBoost

# ML Pipeline design/setup
![image](https://user-images.githubusercontent.com/49780811/169189105-b9e10399-fb16-4651-87f9-7126535435f4.png)


# Installation
### Step 1: Downloading source code
```
git clone https://github.com/tutorialcreation/abtest-mlops.git
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

# The tests from the modularized scripts are run in the following notebooks
* EDA analysis ==> notebooks/EDA.ipynb
* Classical AB test ==> notebooks/AB_Testing.ipynb
* Sequential Test ==> notebooks/Sequential_AB_Testing.ipynb
* Machine learning ==> notebooks/Modeling.ipynb

# Data
The BIO data for this project is a “Yes” and “No” response of online users to the following question
Q: Do you know the brand LUX?

	1. Yes
	2. No

# Dataset Column description
* auction_id: the unique id of the online user who has been presented the BIO. In standard terminologies this is called an impression id. The user may see the BIO questionnaire but choose not to respond. In that case both the yes and no columns are zero.

* experiment: which group the user belongs to - control or exposed.

* date: the date in YYYY-MM-DD format

* hour: the hour of the day in HH format.

* device_make: the name of the type of device the user has e.g. Samsung

* platform_os: the id of the OS the user has.

* browser: the name of the browser the user uses to see the BIO questionnaire.

* yes: 1 if the user chooses the “Yes” radio button for the BIO questionnaire.

* no: 1 if the user chooses the “No” radio button for the BIO questionnaire.

# A/B Hypothesis Testing
A/B testing, also known as split testing, refers to a randomized experimentation process wherein two or more versions of a variable (web page, page element, etc.) are shown to different segments of website visitors at the same time to determine which version leaves the maximum impact and drive business metrics.

## Sequential A/B testing
A common issue with classical A/B-tests, especially when you want to be able to detect small differences, is that the sample size needed can be prohibitively large. In many cases it can take several weeks, months or even years to collect enough data to conclude a test.

* The lower number of errors we require, the larger sample size we need.
* The smaller the difference we want to detect, the larger sample size is required.

Sequential sampling works in a very non-traditional way; instead of a fixed sample size, you choose one item (or a few) at a time, and then test your hypothesis.

