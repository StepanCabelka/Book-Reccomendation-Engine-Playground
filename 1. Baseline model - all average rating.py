import pandas as pd
import numpy as np


'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
ratings_train2=ratings_train.reset_index()
ratings_train_positive=ratings_train[ratings_train.rating>0]

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

def spocti_rating(id,isbn):
    global_dop=avg_rating.loc[isbn,'rating']
    return global_dop

def spocti_predikci():
    df=pd.DataFrame(index=ratings_test.index,columns=['predrating'])
    counter=0
    for id,isbn in df.index:
        counter=counter+1
        print('computing',counter,'of',df.shape[0])
        df.loc[(id,isbn),'predrating']=spocti_rating(id,isbn)
    return df

df=spocti_predikci()
df.to_pickle("D:\\Stepa\\CV\\Datasentics\\baseline recco predikce.pcl")

    
    
    


    