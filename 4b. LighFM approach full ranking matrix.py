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
train user features
'''
features = users_train.reset_index().sort_values('id')
dataset1 = Dataset()
uf = []
col = ['age']*len(features.age.unique()) + ['country']*len(features.country.unique()) + ['town']*len(features.town.unique()) 
unique_f1 = list(features.age.unique()) + list(features.country.unique()) + list(features.town.unique()) 
for x,y in zip(col, unique_f1):
    res = str(x)+ ":" +str(y)
    uf.append(res)

dataset1 = Dataset()
dataset1.fit(users_train.index, books_train.index, user_features = uf)

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
#     print(feature_colon_value(item))
# print(f'Final output: {feature_list}') 

user_tuple = list(zip(features.id, feature_list))
user_features = dataset1.build_user_features(user_tuple, normalize= False)

user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()

model = LightFM(loss='warp',no_components=500)
model.fit(interactions, # spase matrix representing whether user u and item i interacted
      user_features= user_features, # we have built the sparse matrix above
      sample_weight= weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings
      epochs=100,verbose=True)


n_users, n_items = interactions.shape 
l_arr=[]
counter=0
l_col=['rating'+str(i) for i in range(n_items)]
pocet=len(users_test.index)
for id in users_test.index[:]:
    print(counter,'of',pocet)
    user_x = user_id_map[id]
    a=model.predict(user_x, np.arange(n_items))
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
dff['predrating'].to_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf predikce 500.pcl")










# train_auc = auc_score(model,interactions,user_features=user_features).mean()
# test_auc = auc_score(model,interactions_test,user_features=user_features).mean()
# print('Hybrid training set AUC: %s' % train_auc)
# print('Hybrid test set AUC: %s' % test_auc)

# test_predictions=model.predict_rank(interactions_test, train_interactions=interactions, user_features=user_features) 
# df=pd.DataFrame(data=test_predictions.data.astype(int), columns=['predrating'])
# df.predrating.max()
# df.predrating.min()

# test_predictions2=test_predictions.copy()
# test_predictions2.data = np.less(test_predictions2.data, 10, test_predictions2.data)
# precision = np.squeeze(np.array(test_predictions2.sum(axis=1))) / 10
# precision = precision[interactions_test.getnnz(axis=1) > 0]
# precision.mean()






# def Spocti_rating(model):
#     '''predictions for test set'''
#     l_users=[]
#     l_books=[]
#     for (id,isbn) in ratings_test.index:
#         l_users.append(user_id_map[id])
#         l_books.append(item_id_map[isbn])
    
#     test_set_predictions=model.predict(np.array(l_users),np.array(l_books)) 
#     df_test_pred=pd.DataFrame(index=ratings_test.index, data=test_set_predictions, columns=['predrating'])
#     df_test_pred['rating']=ratings_test.rating
#     df_test_pred=df_test_pred.sort_values(by='predrating',ascending=True)

#     '''predictions for train set'''
#     l_users=[]
#     l_books=[]
#     for (id,isbn) in ratings_train.index:
#         l_users.append(user_id_map[id])
#         l_books.append(item_id_map[isbn])
    
#     train_set_predictions=model.predict(np.array(l_users),np.array(l_books)) 
#     df_train_pred=pd.DataFrame(index=ratings_train.index, data=train_set_predictions, columns=['predrating']).sort_values('predrating')

#     '''konvert scores to 0-10 scale
#     konverzi uděláme dle rozložení v train set
#     '''
#     g=ratings_train.groupby('rating')[['rating']].count().sort_index()
#     g['cum']=g['rating'].cumsum()

#     cut_off_points={}
#     for (value, _, cum) in g.itertuples():
#         cut_off_points[value]=df_train_pred.iloc[cum-1,0]

#     df=df_test_pred.copy().sort_values('predrating')
#     df['convrating']=-1
#     for value in range(11):
#         df.loc[(df.predrating<cut_off_points[value]) & (df.convrating==-1),'convrating']=value
#     return df.drop(columns=['predrating','rating']).rename(columns={'convrating':'predrating'})
    
# df=Spocti_rating(model)
# df.to_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf predikce 500.pcl") 









    
    
    









# # dummify categorical features in books
# # books_transformed = pd.get_dummies(books_final, columns = ['published'])
# # bf=list(books_transformed.columns.drop('title'))

# # books_csr = sparse.csr_matrix(books_transformed.drop(columns='title').values)

# # #dummify categorical features in users
# # users_transformed = pd.get_dummies(users_final, columns = ['age', 'country'])
# # users_csr = sparse.csr_matrix(users_transformed.values)
# # uf=list(users_transformed.columns)

# '''dictionaries'''
# user_id = list(users_final.index)
# user_dict = {}
# counter = 0 
# for i in user_id:
#     user_dict[i] = counter
#     counter += 1

# item_dict ={}
# df = books_final.reset_index()[['ISBN', 'title']].sort_values('ISBN').reset_index()
# for i in range(df.shape[0]):
#     item_dict[(df.loc[i,'ISBN'])] = df.loc[i,'title']

# '''
# train test split: no cold start in the test set, i.e. all users in the test must be from the train
# '''
# N=0.1 #pravdepodobnost vyberu konkretniho uzivatele z tech, co maji aspon dve recenze
# g=ratings_final.groupby('id')['rating'].count()
# gg=g[g>1]
# np.random.seed(1)
# l=[]

# for id in gg.index:
#     if np.random.rand()<N:
#         df=ratings_final.loc[(id,),:]
#         n=df.shape[0]
#         a=np.random.randint(n)
#         df2=df.iloc[:a,:]
#         df2['id']=id
#         df2=df2.reset_index()
#         l.append(df2)

# ratings_test=pd.concat(l,axis=0).set_index(['id','ISBN'])
# i=ratings_final.index.difference(ratings_test.index)
# ratings_train=ratings_final.loc[i,:]

# users_test=users_final.loc[ratings_test.index.get_level_values(0).drop_duplicates(),:]
# users_train=users_final.loc[ratings_train.index.get_level_values(0).drop_duplicates(),:]

# books_test=books_final.loc[ratings_test.index.get_level_values(1).drop_duplicates(),:]
# books_train=books_final.loc[ratings_train.index.get_level_values(1).drop_duplicates(),:]

# #train ratings
# l_ratings=[]
# for (id,ISBN),r in ratings_train.itertuples():
#     l_ratings.append((id,ISBN,r))
    
# #all users
# l_users=[]
# for id,age,country,town in users_final.itertuples():
#     l_users.append((id,('age','country','town')))

# #all books
# l_books=[]
# for ISBN in books_final.index:
#     l_books.append((ISBN,('author','published')))

# '''test'''
# l_test_ratings=[]
# for (id,ISBN),r in ratings_test.itertuples():
#     l_test_ratings.append((id,ISBN,r))

# # l_test_users=[]
# # for id,age,country,town in users_test.itertuples():
# #     l_test_users.append((id,["age","country","town"]))

# # l_test_books=[]
# # for ISBN in books_test.index:
# #     l_test_books.append((ISBN,['author','published']))

# '''
# transformace do csr matic
# '''
# from lightfm.data import Dataset

# dataset = Dataset()
# dataset.fit(users=users_final.index, items=books_final.index,user_features=('age','country','town'),item_features=('author','published'))

# train_interactions, train_weights = dataset.build_interactions((i[0], i[1], i[2]) for i in l_ratings)
# test_interactions, test_weight = dataset.build_interactions((i[0], i[1], i[2]) for i in l_test_ratings)
# user_features=dataset.build_user_features(l_users)
# item_features=dataset.build_item_features(l_books)

# # dataset.fit(users=users_test.index, items=books_test.index,user_features=("age",'country','town'),item_features=("author",'published'))
# # test_interactions, test_weights = dataset.build_interactions((i[0], i[1], i[2]) for i in l_test_ratings)
# # user_test_features=dataset.build_user_features(l_test_users)
# # item_test_features=dataset.build_item_features(l_test_books)

# '''
# tréning modelu
# '''

# from lightfm import LightFM
# model = LightFM(loss='warp', random_state=2016, learning_rate=0.90,no_components=150,user_alpha=0.000005)
# model = model.fit(train_interactions, epochs=5,num_threads=16,user_features=user_features,item_features=item_features,
#                   sample_weight=train_weights, verbose=True)

# model2=model.fit(train_interactions, epochs=5,num_threads=16, verbose=True)

# # book_repre=model.get_item_representations()
# # user_repre=model.get_user_representations()


# '''
# ukázka výsledku
# '''
# # predictions = model.predict(
# #     user_ids=np.array([0, 0, 0, 0, 0]),
# #     item_ids=np.array([0, 1, 2, 5, 100]),
# #     item_features=item_features
# # )

# predictions=model.predict(train_interactions.row, train_interactions.col)
# df_predictions=pd.DataFrame(index=ratings_train.index, data=predictions, columns=['rating'])
# df_predictions['rating_orig']=ratings_train.rating
# df_predictions['rating']=df_predictions['rating']/1000000000000

# df_predictions.head(50)

# predictions_test=model.predict(test_interactions.row, test_interactions.col)
# df_predictions_test=pd.DataFrame(index=ratings_test.index, data=predictions_test, columns=['rating'])
# df_predictions_test['rating_orig']=ratings_test.rating


# def sample_recommendation_user(model, interactions, id, threshold = 0,nrec_items = 5, show = True):
    
#     n_users, n_items = interactions.shape
#     user_x = user_dict[id]
#     scores = pd.Series(model.predict(user_x,np.arange(n_items),item_features=books_metadata_csr))
#     scores.index = interactions.columns
#     scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
#     known_items = list(pd.Series(interactions.loc[id,:] \
#                                  [interactions.loc[id,:] > threshold].index).sort_values(ascending=False))
    
#     scores = [x for x in scores if x not in known_items]
#     return_score_list = scores[0:nrec_items]
#     known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
#     scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
#     if show == True:
#         print ("User: " + str(user_id))
#         print("Known Likes:")
#         counter = 1
#         for i in known_items:
#             print(str(counter) + '- ' + i)
#             counter+=1
#             print("\n Recommended Items:")
#         counter = 1
#         for i in scores:
#             print(str(counter) + '- ' + i)
#             counter+=1
            
# a=sample_recommendation_user(model, train_interaction,13026, user_dict, item_dict)    





# '''
# evaluace modelu
# '''

# from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank

# test_interactions, test_weights = dataset.build_interactions((i[0], i[1], i[2]) for i in l_test_ratings)


# pr=precision_at_k(model, test_interactions, k=10, user_features=uf, item_features=bf)
# re=recall_at_k(model, test_interactionsk=10, user_features=user_features, item_features=item_features)
# asc=auc_score(model, test_interactions, user_features=user_features, item_features=item_features)
# asc.shape

# '''
# do model.fit musi vstupovat vsichni useri i knihy. Predict pak musi vyuzivat prevodovych slovniku....
# '''

# model.predict(199,books_test.index)


# '''
# https://github.com/lyst/lightfm/issues/369
# '''


# # dataset.fit_partial(users=np.unique(new_data[:, 0]), items=np.unique(new_data[:, 1]))
# # new_interactions, new_weights = dataset.build_interactions((i[0], i[1], i[2]) for i in new_data)




# # '''
# # zmenšení matice ratingů
# # '''

# # r=ratings_train[ratings_train.rating>0]

# # #dále dropnu usery a knihy, co maji malo ratingu
# # value=True
# # while(value):
# #     g=r.reset_index().groupby('id')[['rating']].count()
# #     zredukuj_users=g[g.rating==1].shape[0]
# #     g=g[g.rating>1]
# #     r=r.reset_index().loc[r.reset_index().id.isin(g.index),:]
# #     gg=r.groupby('ISBN')[['rating']].count()
# #     zredukuj_books=gg[gg.rating==1].shape[0]
# #     print(zredukuj_users,zredukuj_books)
# #     gg=gg[gg.rating>1]
# #     r=r.loc[r.ISBN.isin(gg.index),:].set_index(['id','ISBN'])
# #     if zredukuj_users+zredukuj_books==0:
# #         value=False

# # i_users=r.index.get_level_values(0).drop_duplicates().sort_values()
# # i_books=r.index.get_level_values(1).drop_duplicates().sort_values()
# # users=users_train.loc[users_train.index.isin(i_users),:].sort_index()
# # books=books_train.loc[books_train.index.isin(i_books),:].sort_index()
# # # books.columns

# # user_book_interaction = pd.pivot_table(r, index='id', columns='ISBN', values='rating').fillna(0)
# # user_book_interaction_csr = sparse.csr_matrix(user_book_interaction.values)


# # '''dictionaries'''
# # user_id = list(user_book_interaction.index)
# # user_dict = {}
# # counter = 0 
# # for i in user_id:
# #     user_dict[i] = counter
# #     counter += 1

# # item_dict ={}
# # df = books.reset_index()[['ISBN', 'title']].sort_values('ISBN').reset_index()
# # for i in range(df.shape[0]):
# #     item_dict[(df.loc[i,'ISBN'])] = df.loc[i,'title']


# # # dummify categorical features in books
# # books_transformed = pd.get_dummies(books, columns = ['author', 'published'])
# # books_csr = sparse.csr_matrix(books_transformed.drop(columns='title').values)

# # #dummify categorical features in users
# # users_transformed = pd.get_dummies(users, columns = ['age', 'country','town'])
# # users_csr = sparse.csr_matrix(users_transformed.values)


# # model = lf.LightFM(loss='warp', random_state=2016, learning_rate=0.90,no_components=150,user_alpha=0.000005)
# # model = model.fit(user_book_interaction_csr,epochs=100,num_threads=16, 
# #                   user_features=users_csr,
# #                   item_features=books_csr,verbose=True)




        