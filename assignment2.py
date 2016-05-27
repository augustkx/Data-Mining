#Import relevant packages
import pandas as pd
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from plotbox import plotbox as pb
import random
import itertools
import math
from sklearn.preprocessing import Imputer
import sklearn.ensemble
import pprint
import time
from scipy import stats
import pdb
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
    
def dataload(testfile,trainfile):
    starttime = time.time()
    train = pd.read_csv(trainfile)
    test = pd.read_csv(trainfile) 
    endtime = time.time()
    print("Data loaded in "+str(starttime-endtime)+" seconds.")
    
    #create sampled data
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
    return test, train

# Load data, sample and save
testfile = 'test_set_VU_DM_2014.csv'
trainfile = 'training_set_VU_DM_2014.csv'
#[test, train] = dataload(testfile,trainfile)

#Sample Data and create train, validate and test data from trainingdata
datasam = pd.read_csv('datasam.csv') #load sampled trainingdata
datasam = datasam.drop(datasam.columns[[0]], axis=1)
print("Sampled data loaded")

#Basic information
unique_srch = datasam.srch_id.unique()
unique_hotels = datasam.prop_id.unique()
features = list(datasam.columns.values)
print('The data contains '+str(len(unique_srch))+' unique searches.')
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
    plt.show()
    #plt.savefig('missingvalues.png',bbox_inches='tight')
#plotmissing(nanperc,features)

# Feature Manipulation  
#pdb.set_trace()  #for optional debugging
def createfeat(datasam):
    #Change date variable to year and month
    datasam["date_time"] = pd.to_datetime(datasam["date_time"])
    datasam["year"] = datasam["date_time"].dt.year
    datasam["month"] = datasam["date_time"].dt.month

    #Property location desirability aggregate
    features = list(datasam.columns.values)
    datasam['prop_loc_desir'] = datasam[features[11:13]].mean(axis=1,skipna=True)
    datasam['prop_loc_desir'] = (datasam['prop_loc_desir']/datasam['prop_loc_desir'].max())*10
    
    #score aggregate: starrating and reviewscore
    datasam[features[8:9]] = datasam[features[8:9]]/float(datasam[features[11:12]].max())
    datasam['prop_score'] = datasam['prop_desir'] = datasam[features[8:10]].sum(axis=1,skipna=True, numeric_only = True) #desirablitity score between 1 and 10
    
    #composite of children and adult count
    datasam['srch_person_count'] = (datasam['srch_adults_count']+datasam['srch_children_count'])

    #Composite feature: abs price difference w. competitors
    tempfeats = []
    tempfeats1 = []
    tempfeats2 = []
    for number in range (1,8):
        tempfeats.append('comp'+str(number)+'_rate')
        tempfeats1.append('comp'+str(number)+'_inv')
        tempfeats2.append('comp'+str(number)+'_rate_percent_diff')
    datasam['comp_price'] = datasam[tempfeats].mean(axis=1,skipna=True) 
    datasam['comp_perc_diff'] = datasam[tempfeats2].mean(axis=1, skipna=True)
    datasam['comp_avail'] = datasam[tempfeats1].mean(axis=1,skipna=True) 
    # not price_diff, still too many missing values when combined and only limited correlation
    #impute missing data
    datasam['comp_price'] = datasam.comp_price.fillna(value=-1) #-1 better than median
    datasam['comp_avail'] = datasam.comp_avail.fillna(value=-1) 
    datasam['comp_perc_diff'] = datasam.comp_perc_diff.fillna(value=-1) 
    
    
    #Hotelbooking & clicking correlation with prop_loc_desir and prop_score
    features = list(datasam.columns.values)
    corrframe = datasam[features]
    corrframe = (corrframe[corrframe['click_bool'] ==1])
    corrframe['hotel_clickbool'] = (corrframe['booking_bool']*5)+corrframe['click_bool']
    spearcor=corrframe.corr(method='spearman')["hotel_clickbool"] 
    
    #Impute missing review scores and srch_query_aff score w. median
    median = datasam.prop_review_score.median(axis=0,skipna=True)
    datasam['prop_review_score'] = datasam.prop_review_score.fillna(value=median)
    median = datasam.srch_query_affinity_score.median(axis=0,skipna=True)
    datasam['srch_query_affinity_score'] = datasam.srch_query_affinity_score.fillna(value=median)
    
    #Relevant features for model
    removefeat = features[27:51] #remove competitor data
    removefeat.append('visitor_hist_starrating')
    removefeat.append('visitor_hist_adr_usd')
    removefeat.append('prop_location_score2') #composite added [0-10]
    removefeat.append('prop_location_score1') #composite added [0-10]
    removefeat.append('orig_destination_distance')
    removefeat.append('srch_query_affinity_score')
    removefeat.append('srch_children_count') #badly correlates with booking/clicking
    removefeat.append('date_time') #month and year added
    removefeat.append('year') #year doesn't correlate well at all w. booking/clicking
    removefeat.append('gross_bookings_usd') #remove gross_booking usd
    removefeat.append('position') #training only
    removefeat.append('click_bool') #training only
    removefeat.append('booking_bool') #training only
    newfeat = [feat for feat in features if feat not in removefeat] #create new features 
    
    #Divide into training and test data
    #sampletest = list(set(np.random.choice(len(datasam), int((len(datasam)/2)), replace=False)))
    #sampletrain = list(range(0,len(datasam)))
    #sampletrain = list(set(sampletrain)-set(sampletest))
    #data = datasam
    #test = datasam.ix[sampletest]
    #train = datasam.ix[sampletrain]

    return newfeat, datasam, spearcor #test,train

[newfeat, data, spearcor] = createfeat(datasam)
[ndcg_RF,ndcg_GB,ndcg_SVM] = models(data,newfeat)

#USE FEATURE IMPORTANCES to further add/take out features!!

#%% Run and compare models
def models(data,newfeat):
    predictors= data[newfeat]
    #only 50 samples:
    rows_train = random.sample(list(data['srch_id'].unique()), 50)#select 50 srch_id randomly,used when modeling selecting.
    rows_train = list(data['srch_id'].unique())#used when taining the whole data for the last outcome to hand up.
    rows_test=list(i for i in list(data['srch_id'].unique()) if i not in rows_train)

    #50-50 split of data:
    #rows_test = list(set(np.random.choice(len(datasam), int((len(datasam)/2)), replace=False)))
    #rows_train = list(range(0,len(datasam)))
    #rows_train = list(set(sampletrain)-set(sampletest))
    pd.options.mode.chained_assignment = None #dangerous code, to be refined in time permitted, but I think I can use it here.
    ndcg_randomforest=random_forest(data,data,rows_train,rows_test, predictors)
    print('random forest done: '+str(ndcg_randomforest))
    ndcg_GradientBoosting=GradientBoosting(data,data,rows_train,rows_test, predictors)
    print('GB done: '+str(ndcg_GradientBoosting))
    ndcg_SVM = SVM(data,data,rows_train,rows_test, predictors)
    print('SVM done: '+str(ndcg_SVM))

def ndcg_calc(train_df, pred_scores):
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
     eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
     eval_df['score'] = pred_scores
 
     logger = lambda x: math.log(x + 1, 2)
     eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)
 
     book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
     book_idcg = (31.0 * eval_df['booking_bool']).sum()
     
     click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
     
     # Max number of clicks in training set is 30.
     # Calculate the 31 different contributions to IDCG that 0 to 30 clicks have
     # and put in dict {num of click: IDCG value}.
     disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(31)]
     disc_dict = { i: np.array(disc).cumsum()[i] for i in range(31)}
     
     # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
     # since these were accounted for in book_idcg.
     click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()
 
     return (book_dcg + click_dcg) / (book_idcg + click_idcg)
 
#==============================================================================



#============Random Forest========================================================================================================
def random_forest(data,test,rows_train,rows_test, predictors):
    
    data_test=test[test.srch_id.isin(rows_test)]
    X_test=data_test[predictors]#select the instances, corresponding to the 100 srch_id.
    #===========================predict the booking_bool:==============================================================
    X_train_book=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_book=data[data.srch_id.isin(rows_train)]['booking_bool']
#    Y_test_book=data[data.srch_id.isin(rows_test)]['booking_bool']    
    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.01)
    clf.fit(X_train_book, Y_train_book)
    #scores = cross_validation.cross_val_score(clf, data[predictors], data['click_bool'], cv=3)#no use, we will use the nDCG instead.
    print('random_forest_check1:',1 in clf.predict(X_test),  len(clf.predict(X_test)), len(data_test)    )
    print('random_forest_check2:',np.count_nonzero(clf.predict(X_test)))
#    data_test['book_pre']=clf.predict(data_test.loc[::,predictors])
    data_test['book_pre']=clf.predict(X_test)#Prediction of booking

    print('random_forest_check3:',len(clf.predict(X_test)),len(data_test))
    #===========================predict the click_bool:==============================================================
    X_train_click=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_click=data[data.srch_id.isin(rows_train)]['click_bool']
    clf2 = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.01)
    clf2.fit(X_train_click, Y_train_click)
#    data_test['click_pre']=clf2.predict(X_test_click)
#    data_test['click_pre']=clf.predict(data.loc[rows_test,predictors])[:, 1]
    data_test['click_pre']=clf2.predict(X_test) #Prediction of click
#    print(data_test.head())
     
     #=======================Combine the booking_bool and click_bool to rank the instance grouped by srch_id:====================
    data_test['score'] = data_test['click_pre'].map(lambda x: 1 if x ==1 else 0)#if predicted to be clicked,assign 1 to the score.Else,0.
    data_test.loc[(data_test['book_pre'] ==1),'score']=5#if predicted to be clicked,assign 5 to the score.Else,undo.
#        outcome=data_test['srch_id','prop_id','score']
#        outcome= outcome.sort_values(by=['srch_id','score'], ascending=[True,False])    
#    print(data_test.loc[:,['srch_id','click_bool','booking_bool','click_pre','book_pre','score']])
#    data_test.loc[:,['srch_id','click_bool','booking_bool','click_pre','book_pre','score']].to_csv('check2.csv')
    
    #=======================just for debugging=====================================
    data_test['difference_predict_click']=data_test.click_bool-data_test.click_pre
    data_test['difference_predict_book']=data_test.booking_bool-data_test.book_pre
    print(np.count_nonzero(data_test['difference_predict_click']),np.count_nonzero(data_test['click_bool']),np.count_nonzero(data_test['click_pre']))
    print(np.count_nonzero(data_test['difference_predict_book']),np.count_nonzero(data_test['booking_bool']),np.count_nonzero(data_test['book_pre']))    
    
    
    #=====================nDCG=============================================================================
    if hasattr(data_test, 'booking_bool'):#calculate the nDCG to compare with various models.
#        print('random_forest_ndcg:',ndcg_calc(data_test, data_test['score'])  )
        return ndcg_calc(data_test, data_test['score'])   
    #====================Training outcome==================================================================
    else:#for the last result to hand up.
        outcome=data_test['srch_id','prop_id','score']
        outcome= outcome.sort_values(by=['srch_id','score'], ascending=[True,False])
        return outcome['srch_id','prop_id']

#===========GradientBoostingClassifier=========================================================================================
def GradientBoosting(data,test,rows_train,rows_test, predictors):

    data_test=test[test.srch_id.isin(rows_test)]
    X_test=data_test[predictors]#select the instances, corresponding to the 100 srch_id.
    #===========================predict the booking_bool:==============================================================
    X_train_book=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_book=data[data.srch_id.isin(rows_train)]['booking_bool']
#    Y_test_book=data[data.srch_id.isin(rows_test)]['booking_bool']    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(X_train_book, Y_train_book)
    #clf.score(X_test, y_test)     #no use, we will use the nDCG instead.
    print(' GradientBoosting_check1:',1 in clf.predict(X_test),  len(clf.predict(X_test)), len(data_test)    )
    print(' GradientBoosting_check2:',np.count_nonzero(clf.predict(X_test)))
#    data_test['book_pre']=clf.predict(data_test.loc[::,predictors])
    data_test['book_pre']=clf.predict(X_test)#Prediction of booking

    print(' GradientBoosting_check3:',len(clf.predict(X_test)),len(data_test))
    #===========================predict the click_bool:==============================================================
    X_train_click=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_click=data[data.srch_id.isin(rows_train)]['click_bool']
    clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf2.fit(X_train_click, Y_train_click)
#    data_test['click_pre']=clf2.predict(X_test_click)
#    data_test['click_pre']=clf.predict(data.loc[rows_test,predictors])[:, 1]
    data_test['click_pre']=clf2.predict(X_test) #Prediction of click
#    print(data_test.head())
     
     #=======================Combine the booking_bool and click_bool to get the score:====================
    data_test['score'] = data_test['click_pre'].map(lambda x: 1 if x ==1 else 0)#if predicted to be clicked,assign 1 to the score.Else,0.
    data_test.loc[(data_test['book_pre'] ==1),'score']=5#if predicted to be clicked,assign 5 to the score.Else,undo.
    
#    print(data_test.head())
    
    #=====================nDCG=============================================================================
    if hasattr(data_test, 'booking_bool'):#calculate the nDCG to compare with various models.
#        print(' GradientBoosting_ndcg:',ndcg_calc(data_test, data_test['score'])  )
        return ndcg_calc(data_test, data_test['score'])   
    #====================Training outcome==================================================================
    else:#for the last result to hand up.
        outcome=data_test['srch_id','prop_id','score']
        outcome= outcome.sort_values(by=['srch_id','score'], ascending=[True,False])
        return outcome['srch_id','prop_id']
#=======================================================================================================================


#===========SVM================================================================================================================
def SVM(data,test,rows_train,rows_test, predictors):
    data_test=test[test.srch_id.isin(rows_test)]
    X_test=data_test[predictors]#select the instances, corresponding to the 100 srch_id.
    #===========================predict the booking_bool:==============================================================
    X_train_book=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_book=data[data.srch_id.isin(rows_train)]['booking_bool']
#    Y_test_book=data[data.srch_id.isin(rows_test)]['booking_bool']    
    clf = svm.SVC()
    clf.fit(X_train_book, Y_train_book)
    #clf.score(X_test, y_test)     #no use, we will use the nDCG instead.
    print('SVM_forest_check1:',1 in clf.predict(X_test),  len(clf.predict(X_test)), len(data_test)    )
    print('SVM_check2:',np.count_nonzero(clf.predict(X_test)))
#    data_test['book_pre']=clf.predict(data_test.loc[::,predictors])
    data_test['book_pre']=clf.predict(X_test)#Prediction of booking

    print('SVM_check3:',len(clf.predict(X_test)),len(data_test))
    #===========================predict the click_bool:==============================================================
    X_train_click=data[data.srch_id.isin(rows_train)][predictors]#select the instances, corresponding to the 100 srch_id.
    Y_train_click=data[data.srch_id.isin(rows_train)]['click_bool']
    clf2 = svm.SVC()
    clf2.fit(X_train_click, Y_train_click)
#    data_test['click_pre']=clf2.predict(X_test_click)
#    data_test['click_pre']=clf.predict(data.loc[rows_test,predictors])[:, 1]
    data_test['click_pre']=clf2.predict(X_test) #Prediction of click
#    print(data_test.head())
     
     #=======================Combine the booking_bool and click_bool to get the score:====================
    data_test['score'] = data_test['click_pre'].map(lambda x: 1 if x ==1 else 0)#if predicted to be clicked,assign 1 to the score.Else,0.
    data_test.loc[(data_test['book_pre'] ==1),'score']=5#if predicted to be clicked,assign 5 to the score.Else,undo.
    
#    print(data_test.head())
    
    #=====================nDCG=============================================================================
    if hasattr(data_test, 'booking_bool'):#calculate the nDCG to compare with various models.
#        print('SVM_ndcg:',ndcg_calc(data_test, data_test['score'])  )
        return ndcg_calc(data_test, data_test['score'])   
    #====================Training outcome==================================================================
    else:#for the last result to hand up.
        outcome=data_test['srch_id','prop_id','score']
        outcome= outcome.sort_values(by=['srch_id','score'], ascending=[True,False])
        return outcome['srch_id','prop_id']
#=======================================================================================================================
