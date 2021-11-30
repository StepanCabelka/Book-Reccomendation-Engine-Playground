import pandas as pd
import numpy as np
from scipy import sparse
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score

'''
https://towardsdatascience.com/recommendation-system-in-python-lightfm-61c85010ce17
https://making.lyst.com/lightfm/docs/home.html


Train and tests sets have to have the same dimensionality. 
In your example you have fewer users (30k vs 89k) in the test set, and so 
user 0 in the test set may not map to user 0 in the training set and so on.


https://github.com/lyst/lightfm/issues/155
In the binary case, you can interpret AUC as the probability that a randomly chosen positive 
example will be ranked higher than a randomly chosen negative example. 
Precision at 3 tells you what proportion of the top 3 results are positives. 
In your case where you have far more than 3 items, it's perfectly possible for positives to be ranked 
correctly overall (high AUC), but not make it to the top 3 
(say, putting your positive at position 4 will give you 0.99 AUC but 0 precision@3). 
The results you are getting look sensible to me.
The prediction scores generally do not have an interpretation. They are simply a means of ranking the items.

The scores are local to the individual user, so unfortunately you can't compare scores between users. 
You're correct in what you said -- those scores are only used for ranking items for user 0.
'''

'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
ratings_train2=ratings_train.reset_index()

books_final=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_final.pcl")
users_final=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_final.pcl").sort_index()

'''
nacteni test data
'''
users_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_test.pcl")
books_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_test.pcl")
ratings_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_test.pcl")
ratings_test2=ratings_test.reset_index()

# users_test.index.isin(users_train.index).all()
# books_test.index.isin(books_train.index).all

'''
train user features and item features
'''
features = users_train.reset_index().sort_values('id')
dataset1 = Dataset()
uf = []
col = ['age']*len(features.age.unique()) + ['country']*len(features.country.unique()) + ['town']*len(features.town.unique()) 
unique_f1 = list(features.age.unique()) + list(features.country.unique()) + list(features.town.unique()) 
for x,y in zip(col, unique_f1):
    res = str(x)+ ":" +str(y)
    uf.append(res)


item_features = books_train.reset_index().sort_values('ISBN').drop(columns=['title'])
itf = []
col = ['author']*len(item_features.author.unique()) + ['published']*len(item_features.published.unique())
unique_if1=list(item_features.author.unique()) + list(item_features.published.unique())
for x,y in zip(col, unique_if1):
    res= str(x)+ ":" +str(y)
    itf.append(res)
    

dataset1 = Dataset()
dataset1.fit(users_train.index, books_train.index, user_features = uf,item_features=itf)

(interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in ratings_train2.values])
(interactions_test, weights_test) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in ratings_test2.values])

'''
Building user featues
'''
def feature_colon_value(my_list):
    """
    Takes as input a list and prepends the columns names to respective values in the list.
    For example: if my_list = [1,1,0,'del'],
    resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']
    """
    result = []
    ll = ['age:','country:', 'town:']
    aa = my_list
    for x,y in zip(ll,aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result

ad_subset = features[["age", 'country','town']] 
ad_list = [list(x) for x in ad_subset.values]
feature_list = []
for item in ad_list:
    feature_list.append(feature_colon_value(item))

user_tuple = list(zip(features.id, feature_list))
user_features = dataset1.build_user_features(user_tuple, normalize= False)

'''
Building item features
'''

def feature_item_colon_value(my_list):
    result = []
    ll = ['author:','published:']
    aa = my_list
    for x,y in zip(ll,aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result

ad_subset=item_features[['author','published']]
ad_list = [list(x) for x in ad_subset.values]
feature_list=[]
for item in ad_list:
    feature_list.append(feature_item_colon_value(item))

item_tuple=list(zip(item_features.ISBN,feature_list))
item_features = dataset1.build_item_features(item_tuple, normalize = False)

user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()


model_2features = LightFM(loss='warp',no_components=500)
model_2features.fit(interactions, user_features= user_features, item_features=item_features, sample_weight= weights, 
          epochs=50,verbose=True)


model_2features_800 = LightFM(loss='warp',no_components=800)
model_2features_800.fit(interactions, user_features= user_features, item_features=item_features, sample_weight= weights, 
          epochs=100,verbose=True)




# train_auc = auc_score(model_2features,interactions,user_features=user_features, item_features=item_features).mean()
# test_auc = auc_score(model_2features,interactions_test,user_features=user_features, item_features=item_features).mean()
# print('Hybrid training set AUC: %s' % train_auc)
# print('Hybrid test set AUC: %s' % test_auc)

'''Compute prediction for test users and all books, rescale for 0-10, select only relevant books'''

n_users, n_items = interactions.shape # no of users * no of items
l_arr=[]
counter=0
l_col=['rating'+str(i) for i in range(n_items)]
pocet=len(users_test.index)
for id in users_test.index[:]:
    print(counter,'of',pocet)
    user_x = user_id_map[id]
    a=model_2features_800.predict(user_x, np.arange(n_items))
    a=a-a.min()
    a=a/a.max()*10
    l_arr.append(a)
    counter=counter+1

test_predictions=np.vstack(l_arr)
test_predictions.shape

book_dict={}
for k in item_id_map.keys():
    book_dict[item_id_map[k]]=k

df=pd.DataFrame(index=users_test.index,columns=[book_dict[k] for k in range(n_items)],data=test_predictions)
df.index.name=('id')

predratings=[]
for (id,isbn) in ratings_test.index:
    predratings.append(df.loc[id,isbn])

dff=ratings_test.copy()
dff['predrating']=np.array(predratings)
dff['predrating'].to_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf bookf predikce 800.pcl")

        