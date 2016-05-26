import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
from sklearn import svm
import math
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


#data=pd.read_csv('training_set_VU_DM_2014.csv')
##test=pd.read_csv('test_set_VU_DM_2014.csv')
#print(data.shape)#number of rows&columns:(4958347, 54)
#print(len(data['srch_id'].unique()))#the number of searches:199795
##print(data['srch_id'].unique().max())
#print(len(data['srch_destination_id'].unique()))#the number of searches:18127
#print(len(data["prop_id"].unique()))#129113
##print(data["prop_id"].value_counts())
#==========extract samples:have done and only need done once.================================================
#rows = random.sample(list(data['srch_id'].unique()), 100)#select 100 srch_id randomly
#train=data[data.srch_id.isin(rows)]#select the instances, corresponding to the 100 srch_id.
##np.savetxt("train.csv", train)
#train.to_csv('train.csv')
#==================sample data==================================================================================================
data=pd.read_csv('train.csv')
# print(data.describe())
#print(data.shape)
#data = data.drop(data.columns[[0]], axis=1)
#print("Data loaded")
#print(data.shape)



# convert date_time to numeric values in seperate month and year columns.
data["date_time"] = pd.to_datetime(data["date_time"])
data["year"] = data["date_time"].dt.year
data["month"] = data["date_time"].dt.month
#test["date_time"] = pd.to_datetime(test["date_time"])
#test["year"] = test["date_time"].dt.year
#test["month"] = test["date_time"].dt.month


#==========Feature manipulation function [From Juroen]====================================================================
def new_feat(datasam):
    #composite of children and adult count
    features = list(datasam.columns.values)
    datasam['srch_person_count'] = (datasam['srch_adults_count']+datasam['srch_children_count'])
    
    #Property location desirability aggregate --> NOG FIXEN
    datasam[features[11:12]] = datasam[features[11:12]]/float(datasam[features[11:12]].max())
    datasam['prop_desir'] = datasam['prop_desir'] = datasam[features[11:13]].sum(axis=1,skipna=True, numeric_only = True) #sums values of score1 and score2 (1-10)
    datasam['prop_desir'] = datasam['prop_desir']*5
    
    #score aggregate: starrating and reviewscore
    datasam[features[8:9]] = datasam[features[8:9]]/float(datasam[features[11:12]].max())
    datasam['prop_score'] = datasam['prop_desir'] = datasam[features[8:10]].sum(axis=1,skipna=True, numeric_only = True) #desirablitity score between 1 and 10
    
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
    removefeat.append('year') ##take year out:relevant for test data since it's more recent
    removefeat.append('srch_id') #id
#    removefeat.append('site_id') #id
    removefeat.append('gross_booking_usd') #remove gross_booking usd
    removefeat.append('position') #training only
    removefeat.append('click_bool') #training only
    removefeat.append('booking_bool') #training only
    #create a new features with the new features
    #features = list(datasam.columns.values)
    newfeat = [feat for feat in features if feat not in removefeat] #take feat out
    return newfeat


#==============================just for trying something. Useless.==============================
#print(data.columns)
#print(data.head(5))

#print(data["year"].unique())#year in training:2013,2012
#print(test["year"].unique())#year in testing:2013,1012
##EXPLORATION:plot the distribution of the search ID 
#plt.figure()
#plt.plot(data['srch_id'],'o')
#plt.ylabel('srch_id')
#plt.xlabel('Instances')
#plt.title("srch_id")
#plt.show()
#plt.savefig('srch_id', dpi=200) 
#================================================================================================


#===============================================================================================
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
def random_forest(data,rows_train,rows_test, predictors):
    
    data_test=data[data.srch_id.isin(rows_test)]
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
    print(data_test.head())
     
     #=======================Combine the booking_bool and click_bool to get the score:====================
    data_test['score'] = data_test['click_pre'].map(lambda x: 1 if x ==1 else 0)#if predicted to be clicked,assign 1 to the score.Else,0.
    data_test.loc[(data_test['book_pre'] ==1),'score']=5#if predicted to be clicked,assign 5 to the score.Else,undo.
    
    print(data_test.head())
    
    #=====================nDCG=============================================================================
    if hasattr(data_test, 'booking_bool'):#calculate the nDCG to compare with various models.
#        print('random_forest_ndcg:',ndcg_calc(data_test, data_test['score'])  )
        return ndcg_calc(data_test, data_test['score'])   
    #====================Training outcome==================================================================
    else:#for the last result to hand up.
        outcome=data_test['srch_id','prop_id','score']
        outcome= outcome.sort_values(by=['srch_id','score'], ascending=[True,False])
        return outcome['srch_id','prop_id']
#=======================================================================================================================



#===========GradientBoostingClassifier=========================================================================================
def GradientBoosting(data,rows_train,rows_test, predictors):

    data_test=data[data.srch_id.isin(rows_test)]
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
def SVM(data,rows_train,rows_test, predictors):
    data_test=data[data.srch_id.isin(rows_test)]
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



def main(data):
    """
    run and compare models of random forest, radientBoosting, and SVM.
    """
    #===========Feature selection================================================================
    predictors = [c for c in data.columns if c not in ["srch_id"] and c not in ["date_time"]]
#    predictors = ['prop_country_id','srch_destination_id','prop_id']#,'srch_adults_count','srch_children_count','srch_room_count'
#    predictors= new_feat(data) # [From Juroen]
    
    #print(data[predictors])
    data.fillna(-1, inplace=True)
    rows_train = random.sample(list(data['srch_id'].unique()), 50)#select 50 srch_id randomly,used when modeling selecting.
#    rows_train = list(data['srch_id'].unique())#used when taining the whole data for the last outcome to hand up.
    rows_test=list(i for i in list(data['srch_id'].unique()) if i not in rows_train)
#    rows_test= list(test['srch_id'].unique())#used when taining the whole data for the last outcome to hand up.
#    print(rows_train)
#    print(rows_test)
    pd.options.mode.chained_assignment = None #dangerous code, to be refined in time permitted, but I think I can use it here.
    ndcg_randomforest=random_forest(data,rows_train,rows_test, predictors)
    ndcg_GradientBoosting=GradientBoosting(data,rows_train,rows_test, predictors)
    ndcg_SVM = SVM(data,rows_train,rows_test, predictors)
    print('ndcg_randomforest:',ndcg_randomforest,';   ndcg_GradientBoosting:',ndcg_GradientBoosting,';   ndcg_SVM:', ndcg_SVM)
    print('ndcg_randomforest:',ndcg_randomforest)
    
main(data)





#=========EXPLORATION:PLOTS:exploring the amount of missing data number for each attribute=====================================================================
def missing_data(data):
    print(data.isnull().sum())
    df=data.isnull().sum()#count the nan numbers in each attribute.
    nanperc = (df/len(data))*100#percentage of the missing data in each feature
    plt.figure(figsize=(11,7) )
    plt.gcf().subplots_adjust(bottom=0.4)
    nanperc.plot.bar()
    plt.ylabel('missing')
    plt.xlabel('Attribute')
    plt.title("missing data")
    plt.show()
    plt.savefig('missing data1.png', dpi=200)
    plt.clf()
#missing_data(data)
# ==========EXPLORATION and simple algorithm:the most common prop_ids across the data=============================================================
def most_common_id(data):    
    data_book = data[data['booking_bool'] ==1]
    data_click = data[data['click_bool'] ==1]
    most_common_book = list(data_book .prop_id.value_counts().head(5).index)#to find the most common booked prop_ids across the data
    most_common_click = list(data_click.prop_id.value_counts().head(5).index)#to find the most common clicked prop_ids across the data
    print(most_common_book)#[1535, 130023, 77218, 93348, 44941];whole data[116942, 22578, 77089, 53494, 137997]
    print(most_common_click)#[42200, 1535, 95608, 42564, 104517];whole data[116942, 77089, 22578, 137997, 104517]
    print(data_book .prop_id.value_counts())
    print(data_click .prop_id.value_counts())
    plt.figure(figsize=(10,8))
    
    plt.subplot(211) 
    data_book .prop_id.value_counts().plot.bar()
    plt.title("Strong prop_id bias_clicking" )
    plt.xticks(range(-int(len(data_book['prop_id'].unique())/100),len(data_book['prop_id'].unique()),10))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    
    plt.xlabel('')
    #plt.savefig('position_click basis.png', dpi=200)
    plt.subplot(212) 
    data_click .prop_id.value_counts().plot.bar()
    plt.title("Strong prop_id bias_booking" )
    plt.xlabel('')
    plt.xticks(range(-int(len(data_click['prop_id'].unique())/100),len(data_click['prop_id'].unique()),10))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    
    plt.savefig('booking&clicking ID basis.png', dpi=200)
    plt.clf()
#most_common_id(data)

#=========EXPLORATION:PLOTS:correlation=======================================================
def cor(data):
    cor=data.corr()["prop_id"]
    cor.drop('prop_id',inplace=True)
    print(cor)
    cor.plot(style='o',figsize=(13,5))
    plt.title("Exploring the correlations between prop_id and other features")
    plt.xticks(range(len(data.columns)),cor.index.values,size='small',rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.legend(bbox_to_anchor=(2, 1), loc=2, borderaxespad=0.)
    plt.xlabel('attributes')
    plt.ylabel('correlation')
    plt.savefig('correlation.png', dpi=200)
    plt.clf()
#cor(data)

#=========EXPLORATION:PLOTS:exploring the position basis/Count the number one hotel cluster is clicked and booked====================
def position_basis(data):
    id_position=data.unique()
    df=data['position']
    print(data['position'].value_counts())
    print(data.groupby(by=['position'])['click_bool'].sum())
    
    
    click=data.groupby(by=['position'])['click_bool'].sum().order(ascending=False)
    click.columns = [ 'click']
    booking=data.groupby(by=['position'])['booking_bool'].sum().order(ascending=False)
    booking.columns = [ 'booking']
    plt.figure(figsize=(10,8))
    plt.subplot(211) 
    click.plot.bar()
    plt.title("Strong position bias_clicking" )
    plt.xticks(range(-5,40,10),[0,10,20,30])
    plt.xlabel('')
    #plt.savefig('position_click basis.png', dpi=200)
    plt.subplot(212) 
    booking.plot.bar()
    plt.title("Strong position bias_booking" )
    plt.xlabel('')
    plt.xticks(range(-5,40,10),[0,10,20,30])
    plt.savefig('booking&clicking ID basis.png', dpi=200)
    plt.clf()
#position_basis(data)
    

    
    
    
    
    
    
    