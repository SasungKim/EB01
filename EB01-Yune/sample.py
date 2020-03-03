import pandas as pd
import numpy as np
import csv

from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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

header = ['function','pronoun','ppron','i','we','you','shehe','they','ipron','article','prep',
          'auxverb','adverb','conj','negate','verb','adj','compare','interrog','number','quant','affect',
          'posemo','negemo','anx','anger','sad','social','family','friend','female','male','cogproc',
          'insight','cause','discrep','tentat','certain','differ','percept','see','hear','feel','bio',
          'body','health','sexual','ingest','drives','affiliation','achieve','power','reward','risk',
          'focuspast','focuspresent','focusfuture','relativ','motion','space','time','work','leisure',
          'home','money','relig','death','informal','swear','netspeak','assent','nonflu','filler']

dic = pd.read_csv('LIWC_dictionary.csv', delimiter = ',', names = header,  encoding = 'utf-8-sig')
dic = dic.drop(0)

   
ps = PorterStemmer() 

# choose some words to be stemmed 
function = list(dic['function'].dropna())
pronoun = list(dic['pronoun'].dropna())
ppron = list(dic['ppron'].dropna()) 
i = list(dic['i'].dropna())
we = list(dic['we'].dropna())
you = list(dic['you'].dropna())
shehe = list(dic['shehe'].dropna())
they = list(dic['they'].dropna())
ipron = list(dic['ipron'].dropna())
article = list(dic['article'].dropna())
prep = list(dic['prep'].dropna())
auxverb = list(dic['auxverb'].dropna())
adverb = list(dic['adverb'].dropna())
conj = list(dic['conj'].dropna())
negate = list(dic['negate'].dropna())
verb = list(dic['verb'].dropna())
adj = list(dic['adj'].dropna())
compare = list(dic['compare'].dropna())
interrog = list(dic['interrog'].dropna())
number = list(dic['number'].dropna())
quant = list(dic['quant'].dropna())
affect = list(dic['affect'].dropna())
posemo = list(dic['posemo'].dropna())
negemo = list(dic['negemo'].dropna())
anx = list(dic['anx'].dropna())
anger = list(dic['anger'].dropna())
sad = list(dic['sad'].dropna())
social = list(dic['social'].dropna())
family = list(dic['family'].dropna())
friend = list(dic['friend'].dropna())
female = list(dic['female'].dropna())
male = list(dic['male'].dropna())
cogproc = list(dic['cogproc'].dropna())
insight = list(dic['insight'].dropna())
cause = list(dic['cause'].dropna())
discrep = list(dic['discrep'].dropna())
tentat = list(dic['tentat'].dropna())
certain = list(dic['certain'].dropna())
differ = list(dic['differ'].dropna())
percept = list(dic['percept'].dropna())
see = list(dic['see'].dropna())
hear = list(dic['hear'].dropna())
feel = list(dic['feel'].dropna())
bio = list(dic['bio'].dropna())
body = list(dic['body'].dropna())
health = list(dic['health'].dropna())
sexual = list(dic['sexual'].dropna())
ingest = list(dic['ingest'].dropna())
drives = list(dic['drives'].dropna())
affiliation = list(dic['affiliation'].dropna())
achieve = list(dic['achieve'].dropna())
power = list(dic['power'].dropna())
reward = list(dic['reward'].dropna())
risk = list(dic['risk'].dropna())
focuspast = list(dic['focuspast'].dropna())
focuspresent= list(dic['focuspresent'].dropna())
focusfuture = list(dic['focusfuture'].dropna())
relativ = list(dic['relativ'].dropna())
motion = list(dic['motion'].dropna())
space = list(dic['space'].dropna())
time = list(dic['time'].dropna())
work = list(dic['work'].dropna())
leisure = list(dic['leisure'].dropna())
home = list(dic['home'].dropna())
money = list(dic['money'].dropna())
relig = list(dic['relig'].dropna())
death = list(dic['death'].dropna())
informal = list(dic['informal'].dropna())
swear = list(dic['swear'].dropna())
netspeak = list(dic['netspeak'].dropna())
assent = list(dic['assent'].dropna())
nonflu = list(dic['nonflu'].dropna())
filler = list(dic['filler'].dropna())
count = 0
  
def stemming(list):
   for elements in list:    
    elements = ps.stem(elements)
    return list.append(elements)

w = ps.stem('youve')
print(w)