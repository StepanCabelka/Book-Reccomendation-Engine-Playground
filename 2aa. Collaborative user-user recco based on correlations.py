import pandas as pd
import numpy as np

vysledek=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_korelace.pcl")
users_vysledek=vysledek.index.get_level_values(0).drop_duplicates()
vysledek2=vysledek.reset_index()
'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
ratings_train2=ratings_train.reset_index()

books_final=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_final.pcl")

'''
nacteni test data
'''
users_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_test.pcl")
books_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_test.pcl")
ratings_test=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_test.pcl")
ratings_test2=ratings_test.reset_index()


'''globalni popularita: vcetne knih, co maji nulove hodnoceni'''
avg_rating=ratings_train2.groupby('ISBN')[['rating']].mean().sort_values('rating',ascending=False)
avg_rating.to_pickle("D:\\Stepa\\CV\\Datasentics\\average_rating.pcl")


# id=16
# korelace=vysledek.loc[(16,),:]
# suma=(korelace.sum()+0.001).values
# relevant_ids=korelace.index.sort_values()
# r=ratings_train.reset_index()
# relevant_ratings=r.reset_index().loc[(r.id.isin(relevant_ids)) & (r.rating>0),:].set_index(['id','ISBN']).drop('index',1)
# relevant_books=relevant_ratings.index.get_level_values(1).unique().sort_values()
# matrix=np.zeros((len(relevant_ids),len(relevant_books)),dtype='int16')
# for i in range(len(relevant_ids)):
#     for b in range(len(relevant_books)):
#         t=(relevant_ids[i],relevant_books[b])
#         if t in (relevant_ratings.index): 
#             matrix[i,b]=relevant_ratings.loc[t,'rating']
# matrix2=np.multiply(matrix,korelace.values)
# dopo=matrix2.sum(axis=0)
# dopo=dopo*1/suma
# dopo_dict={relevant_books[i]:dopo[i] for i in range(dopo.shape[0])}
# dopo_dict = {k: v for k, v in sorted(dopo_dict.items(), key=lambda item: item[1], reverse=True)}
# df=pd.DataFrame(index=dopo_dict.keys(),data=dopo_dict.values(),columns=['rating'])
# df.index.name='ISBN'

# def spocti_doporuceni(id,how_many):
#     korelace=vysledek.loc[(id,),:]
#     suma=(korelace.sum()+0.001).values
#     relevant_ids=korelace.index.sort_values()
#     r=ratings_train.reset_index()
#     relevant_ratings=r.reset_index().loc[(r.id.isin(relevant_ids)) & (r.rating>0),:].set_index(['id','ISBN']).drop('index',1)
#     relevant_books=relevant_ratings.index.get_level_values(1).unique().sort_values()
#     matrix=np.zeros((len(relevant_ids),len(relevant_books)),dtype='int16')
#     for i in range(len(relevant_ids)):
#         for b in range(len(relevant_books)):
#             t=(relevant_ids[i],relevant_books[b])
#             if t in (relevant_ratings.index): 
#                 matrix[i,b]=relevant_ratings.loc[t,'rating']
#     matrix2=np.multiply(matrix,korelace.values)
#     dopo=matrix2.sum(axis=0)
#     dopo=dopo*1/suma
#     dopo_dict={relevant_books[i]:dopo[i] for i in range(dopo.shape[0])}
#     dopo_dict = {k: v for k, v in sorted(dopo_dict.items(), key=lambda item: item[1], reverse=True)}
#     df=pd.DataFrame(index=dopo_dict.keys(),data=dopo_dict.values(),columns=['rating'])
#     df.index.name='ISBN'
#     df=df[df.rating>0]
#     if df.shape[0]<how_many:
#         avg_r=avg_rating[~avg_rating.index.isin(df.index)]*0.5
#         df=pd.concat([df,avg_r],axis=0).iloc[:how_many,:].sort_values('rating',ascending=False)
#     return df


# spocti_doporuceni(8,8400)

id=278854
def spocti_doporuceni_positive_cor(id,how_many):
    if id in users_vysledek:
        korelace=vysledek.loc[(id,),:]
        korelace=korelace.loc[korelace.korelace>0,:].sort_index()
        relevant_ids=korelace.index
        relevant_ratings=ratings_train2.loc[(ratings_train2.id.isin(relevant_ids)) & (ratings_train2.rating>0),:]
        relevant_books=relevant_ratings.ISBN.unique().sort()
        matrix=relevant_ratings.pivot(index='id',columns='ISBN',values='rating').sort_index()
        vahy=np.repeat(korelace.values.reshape(matrix.shape[0],1),matrix.shape[1]).reshape(matrix.shape)*(~matrix.isna())
        vahy_sum=vahy.sum(axis=0)
        df=pd.DataFrame((matrix*vahy).sum(axis=0)/vahy_sum).rename(columns={0:'rating'}).sort_values('rating',ascending=False)
        df['typ']='individual'
        df=df.iloc[:how_many,:]
    else:
        df=pd.DataFrame(columns=['rating'])
    if df.shape[0]<how_many:
        avg_r=avg_rating[~avg_rating.index.isin(df.index)]*0.5
        avg_r=avg_r.iloc[:how_many-df.shape[0],:]
        avg_r['typ']='global'
        df=pd.concat([df,avg_r],axis=0).iloc[:how_many,:].sort_values('rating',ascending=False)
    df['title']=books_final.loc[df.index,'title']
    return df

def spocti_rating(id,isbn):
    global_dop=avg_rating.loc[isbn,'rating']
    if id in users_vysledek:
        korelace=vysledek.loc[(id,),:]
        korelace=korelace.loc[korelace.korelace>0,:].sort_index()
        relevant_ids=korelace.index
        relevant_ratings=ratings_train2.loc[(ratings_train2.id.isin(relevant_ids)) & (ratings_train2.rating>0),:]
        relevant_books=relevant_ratings.ISBN.unique()
        if isbn in relevant_books:
            matrix=relevant_ratings.pivot(index='id',columns='ISBN',values='rating').sort_index()
            vahy=np.repeat(korelace.values.reshape(matrix.shape[0],1),matrix.shape[1]).reshape(matrix.shape)*(~matrix.isna())
            vahy_sum=vahy.sum(axis=0)
            df=pd.DataFrame((matrix*vahy).sum(axis=0)/vahy_sum).rename(columns={0:'rating'}).sort_values('rating',ascending=False)
            return df.loc[isbn,'rating']
        else:
            return global_dop
    else:
        return global_dop

# spocti_rating(178,'0345425294')
           

# '''
# rucni kontrola: najdeme usera, ktery ma jen jednoho dalsiho jemu podobneho
# '''
# g=vysledek[vysledek.korelace>0].reset_index().groupby('zdroj_id')['zdroj_id'].count()

# vysledek.loc[278773,:]
# - jen user 92810
# relevant_ratings=ratings_train.loc[([92810],),:]
# relevant_ratings=relevant_ratings[relevant_ratings.rating>0].sort_values('rating',ascending=False)

# a=spocti_doporuceni_positive_cor(278773,139)
# a.drop(columns={'title'})


# '''
# rucni kontrola: najdeme usera, ktery ma jen dva a ty jsou kratke
# '''
# for id in vysledek.index.get_level_values(0).drop_duplicates():
#     df=vysledek.loc[(id,),:]
#     df=df[df.korelace>0]
#     if df.shape[0]==2:
#         df2=ratings_train2.loc[ratings_train2.id.isin(df.index),:]
#         # print(id,df2.shape[0])
#         if df2.shape[0]<=40:
#             print(id)
#             break
# a=spocti_doporuceni_positive_cor(10502,15)[['rating','typ']]
# vysledek.loc[10502,:]
# 162980 - 1
# 226003 - 0.5
# df_162980=ratings_train2.loc[(ratings_train2.id==162980) & (ratings_train2.rating>0),['ISBN','rating']].set_index('ISBN')
# df_226003=ratings_train2.loc[(ratings_train2.id==226003) & (ratings_train2.rating>0),['ISBN','rating']].set_index('ISBN').rename(columns={'rating':'rating2'})

# output=pd.concat([df_162980,df_226003],axis=1).fillna(0)
# output['rating3']=(output['rating']*1 + output['rating2']*0.5)/1.5 
# aa=output.sort_values('rating3',ascending=False)['rating3']

# pd.concat([a,aa],axis=1)

def spocti_predikci():
    df=pd.DataFrame(index=ratings_test.index,columns=['predrating'])
    counter=0
    for id,isbn in df.index:
        # print(id,isbn)
        counter=counter+1
        print('computing',counter,'of',df.shape[0])
        df.loc[(id,isbn),'predrating']=spocti_rating(id,isbn)
    return df

df=spocti_predikci()
df.to_pickle("D:\\Stepa\\CV\\Datasentics\\user-user collaborative recco predikce.pcl")   