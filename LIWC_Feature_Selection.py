import requests
import numpy as np
import pandas as pd
import csv

# For Individual Feature Extraction 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from collections import Counter

# For Relational Feature Extraction
from statistics import mean
import math

# For feature analyzation
#enable multiple outputs per cell
from IPython.core.interactiveshell import InteractiveShell

# increase size of output window
from IPython.core.display import display, HTML

# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# For machine learning
from sklearn import svm
from sklearn.svm import SVC

# Modeling
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# Saving
import joblib
from joblib import dump, load

from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

header = ['Unnamed: 0', 'Statement', 'Label','function','pronoun','ppron','i','we','you','shehe','they','ipron','article','prep',
          'auxverb','adverb','conj','negate','verb','adj','compare','interrog','number','quant','affect',
          'posemo','negemo','anx','anger','sad','social','family','friend','female','male','cogproc',
          'insight','cause','discrep','tentat','certain','differ','percept','see','hear','feel','bio',
          'body','health','sexual','ingest','drives','affiliation','achieve','power','reward','risk',
          'focuspast','focuspresent','focusfuture','relativ','motion','space','time','work','leisure',
          'home','money','relig','death','informal','swear','netspeak','assent','nonflu','filler']

train = pd.read_csv('test_feature.csv', delimiter = ',', names = header,  encoding = 'utf-8-sig')
train = train.drop(0)
test = pd.read_csv('test_feature.csv', delimiter = ',', names = header,  encoding = 'utf-8-sig')
test = test.drop(0)
valid = pd.read_csv('valid_feature.csv', delimiter = ',', names = header,  encoding = 'utf-8-sig')
valid = valid.drop(0)

k=15
cols = [col for col in train.columns if col not in ['Unnamed: 0', 'Label', 'Statement']]


#--------------------------------------
#--------Univariate Selection----------
#--------------------------------------

X = train[cols]
Y = train.Label

LE = preprocessing.LabelEncoder()
train['Label'] = LE.fit_transform(train['Label'])
test['Label'] = LE.fit_transform(test['Label'])
valid['Label'] = LE.fit_transform(valid['Label'])


bestfeatures = SelectKBest(score_func=chi2, k=15)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))
features = pd.DataFrame(featureScores.nlargest(15,'Score'))
features_list = features.Features.tolist()


new_cols = [col for col in train.columns if col in [features_list[0],features_list[1],features_list[2],features_list[3],
                                                    features_list[4],features_list[5],features_list[6],features_list[7],
                                                    features_list[8],features_list[9],features_list[10],features_list[11],
                                                    features_list[12],features_list[13],features_list[14]]]

LE = preprocessing.LabelEncoder()

X_train = train[new_cols]
X_train = preprocessing.scale(X_train)
Y_train = train['Label']


X_test = test[new_cols]
X_test = preprocessing.scale(X_test)
Y_test = test['Label']


X_valid = valid[new_cols]
X_valid = preprocessing.scale(X_valid)
Y_valid = valid['Label']



#Predict on test.csv
kn = KNeighborsClassifier(n_neighbors = 3)
pred_kn = kn.fit(X_train, Y_train).predict(X_test)

KN_test = ClassificationReport(kn, classes=['Fake', 'Not Fake'])
KN_test.fit(X_train, Y_train)
KN_test.score(X_test, Y_test)
KN_test.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_test, pred_kn))
#print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))

#Predict on valid.csv
pred_kn = kn.fit(X_train, Y_train).predict(X_valid)
KN_valid = ClassificationReport(kn, classes = ['Fake', 'Not Fake'])
KN_valid.fit(X_train, Y_train)
KN_valid.score(X_valid, Y_valid)
KN_valid.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_valid, pred_kn))
#print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))


#Naives-Bayes Classification
pipeline = make_pipeline(preprocessing.StandardScaler(), GaussianNB(priors=None))
pred_nb = pipeline.fit(X_train, Y_train).predict(X_test)
NB_test = ClassificationReport(pipeline, classes=['Fake', 'Not Fake'])
NB_test.fit(X_train, Y_train)
NB_test.score(X_test, Y_test)
NB_test.show()

print("NB Accuracy: ", accuracy_score(Y_test, pred_nb))

NB_valid = ClassificationReport(pipeline, classes = ['Fake', 'Not Fake'])
NB_valid.fit(X_train, Y_train)
NB_valid.score(X_valid, Y_valid)
NB_valid.show()

print("NB Accuracy: ", accuracy_score(Y_test, pred_nb))
#print("NB F1-Score: ", f1_score(Y_test, pred_nb))


#--------------------------------------
#--------Feature Importance------------
#--------------------------------------
'''
X = train[cols]
y = train.Label
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
features = feat_importances.nlargest(15)
features.to_csv('selected_features.csv')
features = pd.read_csv('selected_features.csv',delimiter = ',', names = ['Features','Importances'],encoding = 'utf-8-sig')
features_list = features.Features.tolist()

new_cols = [col for col in train.columns if col in [features_list[0],features_list[1],features_list[2],features_list[3],
                                                    features_list[4],features_list[5],features_list[6],features_list[7],
                                                    features_list[8],features_list[9],features_list[10],features_list[11],
                                                    features_list[12],features_list[13],features_list[14]]]

X_train = train[new_cols]
X_train = preprocessing.normalize(X_train)
Y_train = train['Label']

X_test = test[new_cols]
X_test = preprocessing.normalize(X_test)
Y_test = test['Label']

X_valid = valid[new_cols]
X_valid = preprocessing.normalize(X_valid)
Y_valid = valid['Label']

#Predict on test.csv
kn = KNeighborsClassifier(n_neighbors = 3)
pred_kn = kn.fit(X_train, Y_train).predict(X_test)

KN_test = ClassificationReport(kn, classes=['Fake', 'Not Fake'])
KN_test.fit(X_train, Y_train)
KN_test.score(X_test, Y_test)
KN_test.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_test, pred_kn))
#print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))

#Predict on valid.csv
pred_kn = kn.fit(X_train, Y_train).predict(X_valid)
KN_valid = ClassificationReport(kn, classes = ['Fake', 'Not Fake'])
KN_valid.fit(X_train, Y_train)
KN_valid.score(X_valid, Y_valid)
KN_valid.show()

print("KNeighbors Accuracy: ", accuracy_score(Y_valid, pred_kn))
#print("KNeighbors F1-Score: ", f1_score(Y_test, pred_kn))
'''
#--------------------------------------
#--------Correlation Heatmap-----------
#--------------------------------------
'''
k=15
X = train[cols]
y = train.Label
print(y)
#get correlations of each features in dataset
corrmat = train.corr()
top_corr_features = corrmat.nlargest(k, 'Label')['Label'].index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
'''
'''
train_corrmat = train.corr()
train_cols = train_corrmat.nlargest(k, 'Label')['Label'].index
train_chosen_features = pd.concat([train['Statement'], train['Label'], train[train_cols]], axis = 1)

test_corrmat = test.corr()
test_cols = test_corrmat.nlargest(k, 'Label')['Label'].index
test_chosen_features = pd.concat([test['Statement'], test['Label'], test[test_cols]], axis = 1)

valid_corrmat = valid.corr()
valid_cols = valid_corrmat.nlargest(k, 'Label')['Label'].index
valid_chosen_features = pd.concat([valid['Statement'], valid['Label'], valid[valid_cols]], axis = 1)

train_chosen_features = shuffle(train_chosen_features, random_state = 100)
test_chosen_features = shuffle(test_chosen_features, random_state = 100)
valid_chosen_features = shuffle(valid_chosen_features, random_state = 100)

#y = chosen_features['Label'][:160]
#x = chosen_features[cols.drop('Label')][:160]

x_train = train_chosen_features[train_cols.drop('Label')][161:]
y_train = train_chosen_features['Label'][161:]

x_test = test_chosen_features[test_cols.drop('Label')][161:]
y_test = test_chosen_features['Label'][161:]

 
x_valid = valid_chosen_features[valid_cols.drop('Label')][161:]
y_valid = valid_chosen_features['Label'][161:]

#kf = KFold(n_splits = 10, shuffle = False)
#kf.get_n_splits(x)

acc_kn = []
acc_nb = []
'''
