# EB01
Ryerson 2020 Computer Engineering Capstone Project - Fake News Prediction

https://www.ee.ryerson.ca/capstone/topics/2019/EB01.html

Background
The purpose of this notebook is to train a binary classification model based on the liar dataset [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip], that determines whether a statement snipped from a news article is true news or false news (misinformation or disinformation).

Our model features were selected from the feature list of current fake news research [https://arxiv.org/pdf/1812.00315.pdf], feature selection was performed and multiples models were tested to select the one with the best accuracy score.

Our team of four set out to:

- familiarize ourselves with current research on fake news prediction
- design features that can improve prediction accuracy of current models
- implement machine learning models that can take text input and output prediction results

## Getting Started
- clone current repository and run the .ipynb file in jupyter notebook
- data-exploration takes input dataset, and outputs graphical and numerical analysis results on the dataset values within notebook
- data-cleaning takes input dataset(csv), and returns cleaned input data as dataframe, also outputs (csv)
- feature-extraction takes in cleaned input data(csv), and returns extracted feature sets as dataframe, also outputs(csv)
- feature-selection takes in feature set(csv) and returns selected feature set dataframe, based on set criterias that reduces dimensionality and improves performance of potential models, also outputs csv
- model-building is the top-level script that takes in input raw dataset(csv), and generate a trained model along with tested accuracy results shown within notebook (joblib)
- model-testing takes in input text content (string) and outputs the true/false status of the text content based on trained model within notebook

### Prerequisites
- Python 3
- Jupyter Notebook

## Acknowledgements
- FLC: Dr. E. Bagheri

