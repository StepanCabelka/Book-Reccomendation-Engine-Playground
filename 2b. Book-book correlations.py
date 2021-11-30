from scipy.sparse import lil_matrix
#from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
import pandas as pd
import numpy as np
from scipy import sparse
import numpy.ma as ma

'''
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?
'''

'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").reset_index().set_index(['ISBN','id']).sort_index()
ratings_train['rating']=ratings_train['rating'].astype('int8')
ratings_train2=ratings_train.reset_index()

'''
correlation between book isbn and the rest of books
'''
g=ratings_train.groupby('ISBN')[['rating']].count()
g.sort_values('rating')
g[g.rating>1]


def priprav_matici(isbn):
    '''
    pripravi matici, ze ktere budou pocitany korelace
    '''
    i_id=ratings_train.loc[(isbn,),:].index
    relevant_books=ratings_train2.loc[ratings_train2.id.isin(i_id),['ISBN']].set_index('ISBN').sort_index().index.drop_duplicates()
    # df_corr=pd.DataFrame(index=i_id,columns=relevant_books,dtype='int8')
    
    a=ratings_train2.loc[(ratings_train2.ISBN.isin(relevant_books)) & (ratings_train2.id.isin(i_id))]
    df_corr=a.pivot(index = 'id', columns ='ISBN', values = 'rating')
    '''prehodim sloupce'''
    l=[isbn] + list(df_corr.columns.drop(isbn))
    df_corr=df_corr.loc[:,l]
    
    '''dropnu vsechny sloupce az na prvni, co maji stejne minimum i maximum, protoze pak vyjde korelace NA'''
    mask=~(df_corr.min(axis=0)==df_corr.max(axis=0))
    mask[isbn]=True
    df_corr=df_corr.loc[:,mask]
    
    '''dropnu vsechny sloupce co maji prekryv s prvnim jen v jednom ratingu'''
    mask=~df_corr.isna().values
    first=mask[:,0].reshape(-1,1)
    mask_first=np.repeat(first,mask.shape[1],axis=1)
    
    mask_vysledek=((mask_first & mask).sum(axis=0)>1)
    df_corr=df_corr.loc[:,mask_vysledek]
    return df_corr

a=priprav_matici('0002005018')
a.corr()['0002005018']

def spocti_korelaci(matice):
    isbn=matice.columns[0]
    output=matice.corr()[isbn].reset_index()
    output['ISBN_zdroj']=isbn
    return output.set_index(['ISBN_zdroj','ISBN']).rename(columns={isbn:'korelace'})
    
# b=spocti_korelaci(a)

# TRESHOLD=1000
# def spocti_korelaci(matice,treshold):
#     isbn=matice.columns[0]
#     hranice=matice.shape[1]//treshold*treshold
#     # print("hranice",hranice)
#     if hranice>treshold:
#         vysledek=pd.DataFrame()
#         for t in range(treshold,matice.shape[1],treshold):
#             l=[0]+[i for i in range(t-treshold,t) if i!=0]
#             print("sloupce",0,t-treshold,t-1)
#             maly=matice.iloc[:,l]
#             print(maly.shape)
#             df=maly.corr()[isbn]
#             vysledek=pd.concat([vysledek,df],axis=0)
#             # vysledek['zdroj_id']=id
#              # vysledek=vysledek.reset_index(drop=False).set_index(['zdroj_id','id'])
#         maly=pd.concat([matice.iloc[:,0],matice.iloc[:,t:]],axis=1)
#         print(t,matice.shape[1])
#         df=maly.corr()[isbn]        
#         vysledek=pd.concat([vysledek,df],axis=0)        
#         vysledek['zdroj_isbn']=isbn
#         vysledek=vysledek.reset_index(drop=False).rename(columns={0:'korelace','index':'isbn'})
#         vysledek=vysledek.set_index(['zdroj_isbn','isbn'])
#         ii=~(vysledek.index==(isbn,isbn))
#         return vysledek.loc[ii,:].sort_index()
#     else:
#         vysledek=matice.corr()[isbn].reset_index().rename(columns={isbn:'korelace','index':'isbn'})
#         vysledek.loc[:,'zdroj_isbn']=isbn
#         return vysledek.iloc[1:,:].set_index(['zdroj_isbn','isbn']).sort_index()

# b=spocti_korelaci(a,1000)


def spocti_korelace_pro_vsechny_knihy():
    ii=g[g.rating>1].index
    pocet=len(ii)
    i=0
    l=[]
    for isbn in ii:
        print("procesing book",i,"/",pocet)
        a=priprav_matici(isbn)
        b=spocti_korelaci(a)
        l.append(b)
        i=i+1
    return l

l=spocti_korelace_pro_vsechny_knihy()

vysledek=pd.concat(l,axis=0)
vysledek=vysledek[~vysledek.korelace.isna()]

vysledek.to_pickle("D:\\Stepa\\CV\\Datasentics\\books_korelace.pcl")


# a=ratings_train2.loc[ratings_train2.ISBN=='0001047973',:]


