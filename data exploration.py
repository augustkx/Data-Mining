import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
from sklearn import svm

data=pd.read_csv('training_set_VU_DM_2014.csv')
test=pd.read_csv('test_set_VU_DM_2014.csv')
print(data.shape)#number of rows&columns:(4958347, 54)
print(len(data['srch_id'].unique()))#the number of searches:199795
#print(data['srch_id'].unique().max())
print(len(data['srch_destination_id'].unique()))#the number of searches:18127
print(len(data["prop_id"].unique()))#129113
#print(data["prop_id"].value_counts())




##==============================================EXPLORATION:plot the distribution of the search ID 
plt.figure()
plt.plot(data['srch_id'],'o')
plt.ylabel('srch_id')
plt.xlabel('Instances')
plt.title("srch_id")
plt.show()
plt.savefig('srch_id', dpi=200) 
plt.clf()

# convert date_time to numeric values in seperate month and year columns.
data["date_time"] = pd.to_datetime(data["date_time"])
data["year"] = data["date_time"].dt.year
data["month"] = data["date_time"].dt.month

print(data.head(5))

test["date_time"] = pd.to_datetime(test["date_time"])
test["year"] = test["date_time"].dt.year
test["month"] = test["date_time"].dt.month
print(data["year"].unique())#year in training:2013,2012
print(test["year"].unique())#year in testing:2013,1012





#==============================================================================
# def ndcg_calc(train_df, pred_scores):
#     """
#     >>ndcg_calc(train_df, pred_scores)
#        train_df: pd.DataFrame with Expedia Columns: 'srch_id', 'booking_bool', 'click_bool'
#        pred_scores: np.Array like vector of scores with length = num. rows in train_df
#        
#     Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
#     If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
#     'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
#     
#     NDCG = DCG / IDCG
#     DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
#     IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
#     
#     """
#     eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
#     eval_df['score'] = pred_scores
# 
#     logger = lambda x: math.log(x + 1, 2)
#     eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)
# 
#     book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
#     book_idcg = (31.0 * eval_df['booking_bool']).sum()
#     
#     click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
#     
#     # Max number of clicks in training set is 30.
#     # Calculate the 31 different contributions to IDCG that 0 to 30 clicks have
#     # and put in dict {num of click: IDCG value}.
#     disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(31)]
#     disc_dict = { i: np.array(disc).cumsum()[i] for i in range(31)}
#     
#     # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
#     # since these were accounted for in book_idcg.
#     click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()
# 
#     return (book_dcg + click_dcg) / (book_idcg + click_idcg)
# 
#==============================================================================



#==========extract samples:have done================================================
#rows = random.sample(list(data['srch_id'].unique()), 100)#select 100 srch_id randomly
#train=data[data.srch_id.isin(rows)]#select the instances, corresponding to the 100 srch_id.
##np.savetxt("train.csv", train)
#train.to_csv('train.csv')
#==================sample data==================================================================================================
#data=pd.read_csv('train.csv')
# print(data.describe())



#=========EXPLORATION:PLOTS:exploring the amount of missing data number for each attribute=====================================================================
print(data.isnull().sum())
df=data.isnull().sum()#count the nan numbers in each attribute.
plt.figure(figsize=(11,7) )
plt.gcf().subplots_adjust(bottom=0.4)
df.plot.bar()
plt.ylabel('missing')
plt.xlabel('Attribute')
plt.title("missing data")
plt.show()
plt.savefig('missing data.png', dpi=200)
plt.clf()
#=========EXPLORATION and simple algorithm:the most common prop_ids across the data=============================================================
data_book = data[data['booking_bool'] ==1]
data_click = data[data['click_bool'] ==1]
most_common_book = list(data_book .prop_id.value_counts().head(5).index)#to find the most common booked prop_ids across the data
most_common_click = list(data_click.prop_id.value_counts().head(5).index)#to find the most common clicked prop_ids across the data
print(most_common_book)#[1535, 130023, 77218, 93348, 44941]whole data[116942, 22578, 77089, 53494, 137997]
print(most_common_click)#[42200, 1535, 95608, 42564, 104517]whole data[116942, 77089, 22578, 137997, 104517]
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
#plt.clf()


#=========EXPLORATION:PLOTS:correlation=======================================================
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



#=========EXPLORATION:PLOTS:exploring the position basis/Count the number one hotel cluster is clicked and booked====================
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
