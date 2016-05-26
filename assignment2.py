# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:49:56 2016
@author: Jeroen
"""

# Import relevant packages
import pandas as pd
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from plotbox import plotbox as pb
import random
import itertools
import matplotlib.ticker as mtick
from itertools import count
import math
from sklearn.preprocessing import Imputer
#import ml_metrics as metrics
import operator
import sklearn.ensemble
import pprint

# Load data, sample and save
"""
#train = pd.read_csv('training_set_VU_DM_2014.csv')
#test = pd.read_csv('test_set_VU_DM_2014.csv') 
#print("Data loaded")
#sampleind = np.random.choice(len(train), int(len(train)/10),replace=False)
#datasam = train.ix[sampleind]
#datasam.to_csv('datasam.csv', encoding='utf-8') #save datasam to load instead of data

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
"""

#Sample Data and create train, validate and test data from trainingdata
datasam = pd.read_csv('datasam.csv') #load sampled trainingdata
datasam = datasam.drop(datasam.columns[[0]], axis=1)
print("Data loaded")

#Change date variable to year and month
datasam["date_time"] = pd.to_datetime(datasam["date_time"])
datasam["year"] = datasam["date_time"].dt.year
datasam["month"] = datasam["date_time"].dt.month

#Basic variables
unique_users = datasam.srch_id.unique()
unique_hotels = datasam.prop_id.unique()
features = list(datasam.columns.values)
print('The data contains '+str(len(unique_users))+' unique users.')
print('The data contains '+str(len(datasam.visitor_location_country_id.unique()))+' unique visitor locations.')
print('The data contains '+str(len(unique_hotels))+' unique hotels.')
print('The data contains '+str(len(datasam.prop_country_id.unique()))+ ' unique property countries.')


#Visualize data --> missing values and outliers
nancount = datasam.isnull().sum() #missing values per feature
nanperc = (nancount/len(datasam))*100
nanperc = (np.asarray(nanperc.tolist())).astype(int) #percentage of missing values per feature

def plotmissing(counts,labels):
    plt.bar(range(len(labels)), counts, align='center')
    plt.xlabel('Features')
    plt.title('Percentage missing values per feature')
    plt.ylabel('Percentage missing')
    plt.xticks(range(len(labels)),labels, size='small',rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    #plt.show()
    plt.savefig('missingvalues.png',bbox_inches='tight')
    
#plotmissing(nanperc,features)
    
#plot outliers
#use plotbox.py
    
#Hotelbooking likelihood --> sum of clicks (or bookings) /count
""""
click_like = {}
book_like = {}
for i in range(0,len(unique_hotels)):
    temp = datasam['prop_id'][datasam['prop_id']==unique_hotels[i]].index.tolist()
    click_like[str(unique_hotels[i])] = sum(datasam['click_bool'].ix[temp])/len(temp)
    book_like[str(unique_hotels[i])] = sum(datasam['booking_bool'].ix[temp])/len(temp)
print('Hotel clicking and booking likelihood calculated.')

#Plot hotel click and book distribution
def likelihoodplot(click_like,book_like):
    clicks = click_like.values()

    fig = plt.figure(1, (7,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(perc, click_like.values())

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.show()  
    
likelihoodplot(click_like,book_like)
"""

#Feature Manipulation  

#Composite feature: abs price difference w. competitors
#tempfeats = []
#for number in range (1,8):
#    tempfeats.append('comp'+str(number)+'_rate_percent_diff')
#comp_nan = datasam[tempfeats].isnull().sum(axis=1)
#datasam['price_comp'] = (datasam[tempfeats].sum(axis=1,skipna=True))/(8-comp_nan)#0 = no data or equal lower&higher
#vreemde waarden, werkt nog niet goed
#_inv klopt niet met beschrijving qua waarden
#datasam['avail_comp'] = datasam[tempfeats2].sum(axis=1,skipna=True) #amount of avail comp, 0 = no data or no avail

#Composite of children and adult count
datasam['srch_person_count'] = (datasam['srch_adults_count']+datasam['srch_children_count'])
#Property location desirability aggregate
datasam['prop_loc_desir'] = datasam[features[11:13]].mean(axis=1,skipna=True)
datasam['prop_loc_desir'] = (datasam['prop_loc_desir']/datasam['prop_loc_desir'].max())*10
#score aggregate: starrating and reviewscore
datasam['prop_score'] = (datasam[features[8:10]].mean(axis=1,skipna=True))*2

#Relevant features for model
removefeat = features[27:51] #remove competitor data
removefeat.append('visitor_hist_starrating')
removefeat.append('visitor_hist_adr_usd')
removefeat.append('prop_location_score2') #composite added [0-10]
removefeat.append('prop_location_score1') #composite added [0-10]
removefeat.append('orig_destination_distance')
removefeat.append('random_bool')
removefeat.append('srch_query_affinity_score')
removefeat.append('srch_children_count') #composite added
removefeat.append('srch_adults_count') #composite added
removefeat.append('prop_review_score') #composite added [0-10]
removefeat.append('prop_starrating') #composite added [0-10]
removefeat.append('date_time') #month and year added
#take year out? may not be relevant for test data since it's more recent
removefeat.append('srch_id') #id
removefeat.append('site_id') #id
removefeat.append('gross_bookings_usd') #remove gross_booking usd
removefeat.append('position') #training only
removefeat.append('click_bool') #training only
removefeat.append('booking_bool') #training only
#create a new features with the new features
features = list(datasam.columns.values)
newfeat = [feat for feat in features if feat not in removefeat] #take feat out

#Divide into training and test data
sampletest = list(set(np.random.choice(len(datasam), int((len(datasam)/2)), replace=False)))
sampletrain = list(range(0,len(datasam)))
sampletrain = list(set(sampletrain)-set(sampletest))
data = datasam
test = datasam.ix[sampletest]
train = datasam.ix[sampletrain]

#USE FEATURE IMPORTANCES to further add/take out features!!

#%% NDCG
def ndcg_calc(train, pred_scores):
    """
    >>ndcg_calc(train_df, pred_scores)
       train_df: pd.DataFrame with Expedia Columns: 'srch_id', 'booking_bool', 'click_bool'
       pred_scores: np.Array like vector of scores with length = num. rows in train_df
       
    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    NDCG = DCG / IDCG
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    # Only need part of the dataset, evaluation dataframe
    eval_df = train[['srch_id', 'booking_bool', 'click_bool']]
    eval_df['score'] = pred_scores

    # Calculate the log2(rank_in_results + 1), group by srch_id
    logger = lambda x: math.log(x + 1, 2)
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    # Calculate booking DCG and IDCG
    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
    book_idcg = (31.0 * eval_df['booking_bool']).sum()
    
    # Calculate clicking DCG
    click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum() # 2**1 - 1.0 = 1.0 point for every click, but don't count if booking is true
    
    # Need to know the max number of clicks in training set, for now put on 50.
    # Calculate the 50 different contributions to IDCG that 0 to 50 clicks have
    # and put in dict {num of click: IDCG value}.
    disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(50)]
    disc_dict = { i: np.array(disc).cumsum()[i] for i in range(50)}

    # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
    # since these were accounted for in book_idcg.
    click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()

    return (book_dcg + click_dcg) / (book_idcg + click_idcg)

# Model
def test_pred_sorted(test, model, cols):
    """
    >>test_pred_sorted(test_df, model, cols, regress_model = False)
    test_df: pd.DataFrame that contains columns listed in cols.
    model: SciKitLearn model used for ranking.
    cols: [list of strs] columns in test_df to send to model.
    regress: BOOL, Regressor Model used, as opposed to Classifier.
    
    Return a pd.DataFrame that contains 'srch_id', and 'property_id' columns such that
    the properties are listed in descending order of their model score within each search.
    
    To save output use: test_out_df.to_csv("FILE OUT", index = False, cols = ['srch_id', 'prop_id'], header = ['SearchId','PropertyId'])
    """
    
    scores = model.predict_proba(test[cols])[:, 1]
    test['sort_score'] = scores
    return (test[['srch_id', 'prop_id', 'sort_score']].sort(columns=['srch_id', 'sort_score'], ascending = [True, False]))


#col_list = ['month','srch_destination_id','year','prop_country_id','srch_length_of_stay','srch_children_count','srch_room_count']
col_list = newfeat

"""
Training set: data[1] -> data[int(len(data)/2)]
Test set: data[int(len(data)/2)] -> data[len(data)]
"""

model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, min_samples_split = 1000)

model.fit(data.loc[1:int(len(data)/2), col_list], data.loc[1:int(len(data)/2), "click_bool"])
feature_scores_pairs = [[model.feature_importances_[i], col_list[i]] for i in range(len(col_list))]

if hasattr(model, 'predict_proba'):
    crossval_pred_arr = model.predict_proba(data.loc[int(len(data)/2):len(data), col_list])[:, 1]
else:
    crossval_pred_arr = model.predict(data.loc[int(len(data)/2):len(data), col_list])

print(crossval_pred_arr)

ndcg = ndcg_calc(data.loc[int(len(data)/2):len(data)], crossval_pred_arr)

#result = test_pred_sorted(test, model, col_list)
result = test_pred_sorted(test, model, newfeat)
print ("NDCG:", ndcg)
print ("Feature Importances:")
pprint.pprint(sorted(feature_scores_pairs, reverse = True))
print(result)



#%% Simple model
