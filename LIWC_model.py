import pandas as pd
import numpy as np
import csv

from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC

#Modelling
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

#Evaluation
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from collections import Counter

#Cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#Saving model
import joblib
from joblib import dump, load

#-------------------------------------------------------
#Replacing Header
#-------------------------------------------------------
with open ('train_text.csv', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    header[0] = "ID"
    header[1] = "Statement"
    header[2] = "Labels"

with open('train_text.csv', newline='') as inFile, open('train_LIWC.csv', 'w') as outfile:
    r = csv.reader(inFile)
    w = csv.writer(outfile)
    next(r, None) 
    w.writerow(header)
    for row in r:
        w.writerow(row)

with open('test_text.csv', newline='') as inFile, open('test_LIWC.csv', 'w') as outfile:
    r = csv.reader(inFile)
    w = csv.writer(outfile)
    next(r, None) 
    w.writerow(header)
    for row in r:
        w.writerow(row)
        
with open('valid_text.csv', newline='') as inFile, open('valid_LIWC.csv', 'w') as outfile:
    r = csv.reader(inFile)
    w = csv.writer(outfile)
    next(r, None)
    w.writerow(header)
    for row in r:
        w.writerow(row)

#-------------------------------------------------------
#Model Training and Testing
#-------------------------------------------------------        
train_model = pd.read_csv('train_LIWC.csv')
test_model = pd.read_csv('test_LIWC.csv')
valid_model = pd.read_csv('valid_LIWC.csv')

#Preprocessing
LE = preprocessing.LabelEncoder()
train_model['Labels'] = LE.fit_transform(train_model['Labels'])
test_model['Labels'] = LE.fit_transform(test_model['Labels'])
valid_model['Labels'] = LE.fit_transform(valid_model['Labels'])

cols = [col for col in train_model.columns if col not in ['Unnamed: 0', 'Labels', 'Statement']]

X_train = train_model[cols]
Y_train = train_model['Labels']

X_test = test_model[cols]
Y_test = test_model['Labels']

X_valid = valid_model[cols]
Y_valid = valid_model['Labels']

#Evaluation based on Naive-Bayes
pipeline = make_pipeline(preprocessing.StandardScaler(), GaussianNB(priors=None))
pred_nb = pipeline.fit(X_train, Y_train).predict(X_test)
NB_test = ClassificationReport(pipeline, classes=['Fake', 'Not Fake'])
NB_test.fit(X_train, Y_train)
NB_test.score(X_test, Y_test)
NB_test.show()

NB_valid = ClassificationReport(pipeline, classes = ['Fake', 'Not Fake'])
NB_valid.fit(X_train, Y_train)
NB_valid.score(X_valid, Y_valid)
NB_valid.show()

print("NB Accuracy: ", accuracy_score(Y_test, pred_nb))
print("NB F1-Score: ", f1_score(Y_test, pred_nb))

#Evaluation based on K-Neighbors Classifier
kn = KNeighborsClassifier(n_neighbors = 3)
pred_kn = kn.fit(X_train, Y_train).predict(X_test)

KN_test = ClassificationReport(kn, classes=['Fake', 'Not Fake'])
KN_test.fit(X_train, Y_train)
KN_test.score(X_test, Y_test)
KN_test.show()

KN_valid = ClassificationReport(kn, classes = ['Fake', 'Not Fake'])
KN_valid.fit(X_train, Y_train)
KN_valid.score(X_valid, Y_valid)
KN_valid.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_test, pred_kn))
print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))