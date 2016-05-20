# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:49:56 2016
@author: Jeroen
"""

# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from plotbox import plotbox as pb
import random
import itertools
from itertools import count
import math as mt

#%% Load data, sample and save
train = pd.read_csv('training_set_VU_DM_2014.csv')
#test = pd.read_csv('test_set_VU_DM_2014.csv') 
sampleind = np.random.choice(len(train), int(len(train)/10),replace=False)
datasam = train.ix[sampleind]
datasam.to_csv('datasam.csv', encoding='utf-8') #save datasam to load instead of data

#check presence of search queries in testdata
test_ids = set(test.srch_id.unique())
train_ids = set(train.srch_id.unique())
intersection_count = len(test_ids & train_ids)
print('The test- and trainingdata have '+str(len(test_ids)/len(test))+'% and '+str(len(train_ids)/len(train))+'% of total unique visitors respectively')
print('The test and training data contain '+str(intersection_count)+' of same users.') #false
print('The training- and testdata contain '+str(len(train.srch_id.unique()))+' and '+str(len(test.srch_id.unique()))+' unique users respectively.')
print('The training- and testdata contain '+str(len(train.visitor_location_country_id.unique()))+' and '+str(len(test.visitor_location_country_id.unique()))+' unique visitor locations respectively.')
test_countries = set(test.visitor_location_country_id.unique())
train_countries = set(train.visitor_location_country_id.unique())
intersection_count = len(test_countries & train_countries)
print('The test and training data contain '+str(intersection_count)+' of the same country IDs.')
print('The training- and testdata contain '+str(len(train.prop_id.unique()))+' and '+str(len(test.prop_id.unique()))+' unique hotels respectively.')
print('The training- and testdata contain '+str(len(train.prop_country_id.unique()))+' and '+str(len(test.prop_country_id.unique()))+' unique property countries respectively.')

#%% Sample Data and create train, validate and test data from trainingdata
#load sampled training data
datasam = pd.read_csv('datasam.csv')
datasam = datasam.drop(datasam.columns[[0]], axis=1)

#change date variable to year and month
datasam["date_time"] = pd.to_datetime(datasam["date_time"])
datasam["year"] = datasam["date_time"].dt.year
datasam["month"] = datasam["date_time"].dt.month

#%%Basic variables
unique_users = datasam.srch_id.unique()
unique_hotels = datasam.prop_id.unique()
features = list(datasam.columns.values)

print('The data contains '+str(len(unique_users))+' unique users.')
print('The data contains '+str(len(datasam.visitor_location_country_id.unique()))+' unique visitor locations.')
print('The data contains '+str(len(unique_hotels))+' unique hotels.')
print('The data contains '+str(len(datasam.prop_country_id.unique()))+ ' unique property countries.')

#%% Visualize data

#display missing values and outliers in sample
nancount = datasam.isnull().sum() #missing values per feature
nanperc = (nancount/len(datasam))*100
nanperc = np.asarray(nanperc.tolist())
nanperc = nanperc.astype(int)

def plotmissing(count,labels):
    plt.bar(range(len(labels)), count, align='center')
    plt.xlabel('Features')
    plt.title('Percentage missing values per feature')
    plt.ylabel('Percentage missing')
    plt.xticks(range(len(labels)),labels, size='small',rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    #plt.show()
    plt.savefig('missingvalues.png',bbox_inches='tight')
    
plotmissing(nanperc,features)

#%% Feature manipulation

#Composite feature: normalized price difference w. competitors
#tempfeats = []
#for number in range (1,8):
#    tempfeats.append('comp'+str(number)+'_rate_percent_diff')
#price_comp = datasam[tempfeats].sum(axis=1,skipna=True)
#comp_nan = datasam[tempfeats].isnull().sum() #doesn't work well yet
#price_comp = price_comp/comp_nan #normalize absolute difference by available comp.
#impute missing data w. median

#Take several features out
removefeat = features[27:51] #remove competitor data
removefeat.append(features[4]) #remove history star rating
removefeat.append(features[5]) #remove history usd
removefeat.append(features[12]) #remove second location score
removefeat.append(features[24]) #remove srchquery aff score
removefeat.append('date_time') #remove date_time
removefeat.append(features[21:23])
removefeat.append('position')
removefeat.append(features[52]) #remove gross_booking usd
removefeat.append('srch_destination_id') #remove srch destination ID and use PCA
newfeat = [feat for feat in features if feat not in removefeat] #take feat out

#Create test and training data
training = datasam[newfeat]
training.fillna(-1, inplace=True) #Impute missing values w/ -1

#Check missing values again
#nancount = training.isnull().sum() #missing values per feature
#nanperc = (nancount/len(datasam))*100
#nanperc = np.asarray(nanperc.tolist())
#nanperc = nanperc.astype(int)
#plotmissing(nanperc,newfeat)

#Divide into training and test data
sampletest = list(set(np.random.choice(len(training), int((len(training)/2)), replace=False)))
sampletrain = list(range(0,len(training)))
sampletrain = list(set(sampletrain)-set(sampletest))
test = training.ix[sampletest]
train = training.ix[sampletrain]



#%% Simple model
