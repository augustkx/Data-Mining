# J.D. Janssen
# https://www.dataquest.io/blog/kaggle-tutorial/


from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import random
import ml_metrics as metrics
import operator
import sklearn.ensemble
import seaborn as sns
import math
import pprint

options.mode.chained_assignment = None  # default='warn'


filename = "training_set_VU_DM_2014.csv" 

# Take a random 20% of the data to train
n = 4958347
s = 1000000 #desired sample size

skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

# load the data
data = read_csv(filename, skiprows=skip)
test = read_csv('test_set_VU_DM_2014.csv')
print("Data loaded")

# convert date_time to numeric values in seperate month and year columns
data["date_time"] = to_datetime(data["date_time"])
data["year"] = data["date_time"].dt.year
data["month"] = data["date_time"].dt.month

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


col_list = ['month','srch_destination_id','year','prop_country_id','srch_length_of_stay','srch_children_count','srch_room_count']


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

result = test_pred_sorted(test, model, col_list)
print ("NDCG:", ndcg)
print ("Feature Importances:")
pprint.pprint(sorted(feature_scores_pairs, reverse = True))
print(result)