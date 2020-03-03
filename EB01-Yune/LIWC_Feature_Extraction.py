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

data = pd.read_csv('valid.csv')
name = 'valid'
data_state = data.statement
data_label = data.label
'''
count = 0
for elements in header:
    count += 1
    print (count)
    #print (elements)
    tmp = dic[elements].dropna()
    #exec (elements + " = 'elements'")
    elements = list(tmp)
'''

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
ps = PorterStemmer()


def stemming(list):
   for element in list:    
     a = ps.stem(element)
     new_list.append(a)
   return new_list
       

def check_common(state, list, count):
    for elements in list:
        for words in state.split():
            if words == elements:
                count += 1
    return count

def tagging_univ(str):
    text = nltk.word_tokenize(str)
    tagged = nltk.pos_tag(text, tagset = 'universal')
    return tagged

def tagging_nuniv(str):
    text = nltk.word_tokenize(str)
    tagged = nltk.pos_tag(text)
    return tagged


#-------------------
#Feature Extraction Starts
#-------------------
stemming(function)
stemming(pronoun)
stemming(ppron)
stemming(i)
stemming(we)
stemming(you)
stemming(shehe)
stemming(they)
stemming(ipron)
stemming(article)
stemming(prep)
stemming(auxverb)
stemming(adverb)
stemming(conj)
stemming(negate)
stemming(verb)
stemming(adj)
stemming(compare)
stemming(interrog)
stemming(number)
stemming(quant)
stemming(affect)
stemming(posemo)
stemming(negemo)
stemming(anx)
stemming(anger)
stemming(sad)
stemming(social)
stemming(family)
stemming(friend)
stemming(female)
stemming(male)
stemming(cogproc)
stemming(insight)
stemming(cause)
stemming(discrep)
stemming(tentat)
stemming(certain)
stemming(differ)
stemming(percept)
stemming(see)
stemming(hear)
stemming(feel)
stemming(bio)
stemming(body)
stemming(health)
stemming(sexual)
stemming(ingest)
stemming(drives)
stemming(affiliation)
stemming(achieve)
stemming(power)
stemming(reward)
stemming(risk)
stemming(focuspast)
stemming(focuspresent)
stemming(focusfuture)
stemming(relativ)
stemming(motion)
stemming(time)
stemming(space)
stemming(work)
stemming(leisure)
stemming(home)
stemming(money)
stemming(relig)
stemming(death)
stemming(informal)
stemming(swear)
stemming(assent)
stemming(netspeak)
stemming(nonflu)
stemming(filler)
function_count = list()
pronoun_count = list()
ppron_count = list() 
i_count = list()
we_count = list()
you_count = list()
shehe_count = list()
they_count = list()
ipron_count = list()
article_count = list()
prep_count = list()
auxverb_count = list()
adverb_count = list()
conj_count = list()
negate_count = list()
verb_count = list()
adj_count = list()
compare_count = list()
interrog_count = list()
number_count = list()
quant_count = list()
affect_count = list()
posemo_count = list()
negemo_count = list()
anx_count = list()
anger_count = list()
sad_count = list()
social_count = list()
family_count = list()
friend_count = list()
female_count = list()
male_count = list()
cogproc_count = list()
insight_count = list()
cause_count = list()
discrep_count = list()
tentat_count = list()
certain_count = list()
differ_count = list()
percept_count = list()
see_count = list()
hear_count = list()
feel_count = list()
bio_count = list()
body_count = list()
health_count = list()
sexual_count = list()
ingest_count = list()
drives_count = list()
affiliation_count = list()
achieve_count = list()
power_count = list()
reward_count = list()
risk_count = list()
focuspast_count = list()
focuspresent_count = list()
focusfuture_count = list()
relativ_count = list()
motion_count = list()
space_count = list()
time_count = list()
work_count = list()
leisure_count = list()
home_count = list()
money_count = list()
relig_count = list()
death_count = list()
informal_count = list()
swear_count = list()
assent_count = list()
netspeak_count = list()
nonflu_count = list()
filler_count = list()
#
##---------------------------------------------
##------------Feature Extraction---------------
##---------------------------------------------
for states in data_state:
    
    tagged = tagging_univ(states)
    tagged_nu = tagging_nuniv(states)
    feature = pd.DataFrame()
    count = 0
    
    count = check_common(states, function, count)
    function_count.append(count)
    count = 0
   
    count = check_common(states, pronoun, count)
    pronoun_count.append(count)
    count = 0
    
    count = check_common(states, ppron, count)
    ppron_count.append(count)
    count = 0
    
    count = check_common(states, i, count)
    i_count.append(count)
    count = 0
    
    count = check_common(states, we, count)
    we_count.append(count)
    count = 0
    
    count = check_common(states, you, count)
    you_count.append(count)
    count = 0
    
    count = check_common(states, shehe, count)
    shehe_count.append(count)
    count = 0
    
    count = check_common(states, they, count)
    they_count.append(count)
    count = 0
    
    count = check_common(states, ipron, count)
    ipron_count.append(count)
    count = 0
    
    count = check_common(states, article, count)
    article_count.append(count)
    count = 0
    
    count = check_common(states, prep, count)
    prep_count.append(count)
    count = 0
    
    count = check_common(states, auxverb, count)
    auxverb_count.append(count)
    count = 0
    
    count = check_common(states, adverb, count)
    adverb_count.append(count)
    count = 0
    
    count = check_common(states, conj, count)
    conj_count.append(count)
    count = 0
    
    count = check_common(states, negate, count)
    negate_count.append(count)
    count = 0
    
    count = check_common(states, verb, count)
    verb_count.append(count)
    count = 0
   
    count = check_common(states, adj, count)
    adj_count.append(count)
    count = 0
 
    count = check_common(states, compare, count)
    compare_count.append(count)
    count = 0
    
    count = check_common(states, interrog, count)
    interrog_count.append(count)
    count = 0
   
    count = check_common(states, number, count)
    number_count.append(count)
    count = 0

    count = check_common(states, quant, count)
    quant_count.append(count)
    count = 0

    count = check_common(states, affect, count)
    affect_count.append(count)
    count = 0
  
    count = check_common(states, posemo, count)
    posemo_count.append(count)
    count = 0

    count = check_common(states, negemo, count)
    negemo_count.append(count)
    count = 0

    count = check_common(states, anx, count)
    anx_count.append(count)
    count = 0

    count = check_common(states, anger, count)
    anger_count.append(count)
    count = 0

    count = check_common(states, sad, count)
    sad_count.append(count)
    count = 0

    count = check_common(states, social, count)
    social_count.append(count)
    count = 0
    
    count = check_common(states, family, count)
    family_count.append(count)
    count = 0

    count = check_common(states, friend, count)
    friend_count.append(count)
    count = 0

    count = check_common(states, female, count)
    female_count.append(count)
    count = 0

    count = check_common(states, male, count)
    male_count.append(count)
    count = 0

    count = check_common(states, cogproc, count)
    cogproc_count.append(count)
    count = 0

    count = check_common(states, insight, count)
    insight_count.append(count)
    count = 0

    count = check_common(states, cause, count)
    cause_count.append(count)
    count = 0

    count = check_common(states, discrep, count)
    discrep_count.append(count)
    count = 0

    count = check_common(states, tentat, count)
    tentat_count.append(count)
    count = 0

    count = check_common(states, certain, count)
    certain_count.append(count)
    count = 0

    count = check_common(states, differ, count)
    differ_count.append(count)
    count = 0

    count = check_common(states, percept, count)
    percept_count.append(count)
    count = 0

    count = check_common(states, see, count)
    see_count.append(count)
    count = 0

    count = check_common(states, hear, count)
    hear_count.append(count)
    count = 0

    count = check_common(states, feel, count)
    feel_count.append(count)
    count = 0

    count = check_common(states, bio, count)
    bio_count.append(count)
    count = 0

    count = check_common(states, body, count)
    body_count.append(count)
    count = 0

    count = check_common(states, health, count)
    health_count.append(count)
    count = 0

    count = check_common(states, sexual, count)
    sexual_count.append(count)
    count = 0

    count = check_common(states, ingest, count)
    ingest_count.append(count)
    count = 0

    count = check_common(states, drives, count)
    drives_count.append(count)
    count = 0

    count = check_common(states, affiliation, count)
    affiliation_count.append(count)
    count = 0

    count = check_common(states, achieve, count)
    achieve_count.append(count)
    count = 0

    count = check_common(states, power, count)
    power_count.append(count)
    count = 0
   
    count = check_common(states, reward, count)
    reward_count.append(count)
    count = 0

    count = check_common(states, risk, count)
    risk_count.append(count)
    count = 0

    count = check_common(states, focuspast, count)
    focuspast_count.append(count)
    count = 0

    count = check_common(states, focuspresent, count)
    focuspresent_count.append(count)
    count = 0

    count = check_common(states, focusfuture, count)
    focusfuture_count.append(count)
    count = 0

    count = check_common(states, relativ, count)
    relativ_count.append(count)
    count = 0

    count = check_common(states, motion, count)
    motion_count.append(count)
    count = 0

    count = check_common(states, space, count)
    space_count.append(count)
    count = 0

    count = check_common(states, time, count)
    time_count.append(count)
    count = 0

    count = check_common(states, work, count)
    work_count.append(count)
    count = 0

    count = check_common(states, leisure, count)
    leisure_count.append(count)
    count = 0

    count = check_common(states, home, count)
    home_count.append(count)
    count = 0

    count = check_common(states, money, count)
    money_count.append(count)
    count = 0

    count = check_common(states, relig, count)
    relig_count.append(count)
    count = 0

    count = check_common(states, death, count)
    death_count.append(count)
    count = 0

    count = check_common(states, informal, count)
    informal_count.append(count)
    count = 0

    count = check_common(states, swear, count)
    swear_count.append(count)
    count = 0

    count = check_common(states, netspeak, count)
    netspeak_count.append(count)
    count = 0

    count = check_common(states, assent, count)
    assent_count.append(count)
    count = 0

    count = check_common(states, nonflu, count)
    nonflu_count.append(count)
    count = 0

    count = check_common(states, filler, count)
    filler_count.append(count)
    count = 0
    
feature['Statement'] = data_state
feature['Label'] = data_label
feature['function'] = function_count
feature['pronoun'] = pronoun_count
feature['ppron'] = ppron_count
feature['i'] = i_count
feature['we'] = we_count
feature['you'] = you_count
feature['shehe'] = shehe_count
feature['they'] = they_count
feature['ipron'] = ipron_count
feature['article'] = article_count
feature['prep'] = prep_count
feature['auxverb'] = auxverb_count
feature['adverb'] = adverb_count
feature['conj'] = conj_count
feature['negate'] = negate_count
feature['verb'] = verb_count
feature['adj'] = adj_count
feature['compare'] = compare_count
feature['interrog'] = interrog_count
feature['number'] = number_count
feature['quant'] = quant_count
feature['affect'] = affect_count
feature['posemo'] = posemo_count
feature['negemo'] = negemo_count
feature['anx'] = anx_count
feature['anger'] = anger_count
feature['sad'] = sad_count 
feature['social'] = social_count 
feature['family'] = family_count
feature['friend'] = friend_count
feature['female'] = female_count 
feature['male'] = male_count
feature['cogproc'] = cogproc_count 
feature['insight'] = insight_count
feature['causet'] = cause_count
feature['discrep'] = discrep_count 
feature['tentat'] = tentat_count
feature['certain'] = certain_count 
feature['differ'] = differ_count
feature['percept'] = percept_count 
feature['see'] = see_count
feature['hear'] = hear_count
feature['feel'] = feel_count
feature['bio'] = bio_count 
feature['body'] = body_count 
feature['health'] = health_count
feature['sexual'] = sexual_count 
feature['ingest'] = ingest_count 
feature['drives_'] = drives_count
feature['affiliation'] = affiliation_count 
feature['achieve'] = achieve_count
feature['power'] = power_count
feature['reward'] = reward_count 
feature['risk'] = risk_count 
feature['focuspast'] = focuspast_count 
feature['focuspresent'] = focuspresent_count 
feature['focusfuture'] = focusfuture_count 
feature['relativ'] = relativ_count 
feature['motion'] = motion_count
feature['space'] = space_count 
feature['time'] = time_count 
feature['work'] = work_count
feature['leisure'] = leisure_count
feature['home'] = home_count 
feature['money'] = money_count 
feature['relig'] = relig_count
feature['death'] = death_count 
feature['informal'] = informal_count
feature['swear'] = swear_count 
feature['assent'] = assent_count 
feature['netspeak'] = netspeak_count 
feature['nonflu'] = nonflu_count 
feature['filler'] = filler_count 

feature.to_csv(name + "_feature.csv")    
