# -*- coding: utf-8 -*-
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

train_model = pd.read_csv('train_feature.csv')
test_model = pd.read_csv('test_feature.csv')
valid_model = pd.read_csv('valid_feature.csv')



#Preprocessing
LE = preprocessing.LabelEncoder()

train_model['Label'] = LE.fit_transform(train_model['Label'])
test_model['Label'] = LE.fit_transform(test_model['Label'])
valid_model['Label'] = LE.fit_transform(valid_model['Label'])

cols = [col for col in train_model.columns if col not in ['Unnamed: 0', 'Label', 'Statement']]

X_train = train_model[cols]
Y_train = train_model['Label']

X_test = test_model[cols]
Y_test = test_model['Label']

X_valid = valid_model[cols]
Y_valid = valid_model['Label']

#Evaluation based on Naive-Bayes
pipeline = make_pipeline(preprocessing.StandardScaler(), GaussianNB(priors=None))
pred_nb = pipeline.fit(X_train, Y_train).predict(X_test)
NB_test = ClassificationReport(pipeline, classes=['Fake', 'Not Fake'])
NB_test.fit(X_train, Y_train)
NB_test.score(X_test, Y_test)
NB_test.show()

print("NB Accuracy: ", accuracy_score(Y_test, pred_nb))
print("NB F1-Score: ", f1_score(Y_test, pred_nb))

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

print("KNeighbors Accuracy: ", accuracy_score(Y_test, pred_kn))
print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))

pred_kn = kn.fit(X_train, Y_train).predict(X_valid)
KN_valid = ClassificationReport(kn, classes = ['Fake', 'Not Fake'])
KN_valid.fit(X_train, Y_train)
KN_valid.score(X_valid, Y_valid)
KN_valid.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_valid, pred_kn))
print("KNeighbors F1-Score: ", f1_score(Y_valid, pred_kn))
