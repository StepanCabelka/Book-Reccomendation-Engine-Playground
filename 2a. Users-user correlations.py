import pandas as pd
import numpy as np

'''
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?
'''

'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
ratings_train2=ratings_train.reset_index()
'''
correlation between user i and the rest of users
'''
g=ratings_train.groupby('id')[['rating']].count()
g.sort_values('rating')
g[g.rating>1]

# id=11676

def priprav_matici(id):
    '''
    pripravi matici, ze ktere budou pocitany korelace
    '''
    i_isbn=ratings_train.loc[(id,),:].index.sort_values()
    relevant_users=ratings_train2.loc[ratings_train2.ISBN.isin(i_isbn),'id'].sort_values().unique()
    
    a=ratings_train2.loc[(ratings_train2.ISBN.isin(i_isbn)) & (ratings_train2.id.isin(relevant_users))]
    df_corr=a.pivot(index = 'ISBN', columns ='id', values = 'rating')
    
    '''prehodim sloupce'''
    l=[id] + list(df_corr.columns.drop(id))
    df_corr=df_corr.loc[:,l]
    
    '''dropnu vsechny sloupce az na prvni, co maji stejne minimum i maximum, protoze pak vyjde korelace NA'''
    mask=~(df_corr.min(axis=0)==df_corr.max(axis=0))
    mask[id]=True
    df_corr=df_corr.loc[:,mask]
    
    '''dropnu vsechny sloupce co maji prekryv s prvnim jen v jednom ratingu'''
    mask=~df_corr.isna().values
    first=mask[:,0].reshape(-1,1)
    mask_first=np.repeat(first,mask.shape[1],axis=1)
    
    mask_vysledek=((mask_first & mask).sum(axis=0)>1)
    df_corr=df_corr.loc[:,mask_vysledek]
    return df_corr

# a=priprav_matici(278851)
# b=priprav_matici(11676)

# TRESHOLD=1000

def spocti_korelaci(matice,treshold):
    id=matice.columns[0]
    hranice=matice.shape[1]//treshold*treshold
    # print("hranice",hranice)
    if hranice>treshold:
        vysledek=pd.DataFrame()
        for t in range(treshold,matice.shape[1],treshold):
            l=[0]+[i for i in range(t-treshold,t) if i!=0]
            print("sloupce",0,t-treshold,t-1)
            maly=matice.iloc[:,l]
            print(maly.shape)
            df=maly.corr()[id]
            vysledek=pd.concat([vysledek,df],axis=0)
            # vysledek['zdroj_id']=id
             # vysledek=vysledek.reset_index(drop=False).set_index(['zdroj_id','id'])
        maly=pd.concat([matice.iloc[:,0],matice.iloc[:,t:]],axis=1)
        print(t,matice.shape[1])
        df=maly.corr()[id]        
        vysledek=pd.concat([vysledek,df],axis=0)        
        vysledek['zdroj_id']=id
        vysledek=vysledek.reset_index(drop=False).rename(columns={0:'korelace','index':'id'})
        vysledek=vysledek.set_index(['zdroj_id','id'])
        ii=~(vysledek.index==(id,id))
        return vysledek.loc[ii,:].sort_index()
    else:
        vysledek=matice.corr()[id].reset_index().rename(columns={id:'korelace','index':'id'})
        vysledek.loc[:,'zdroj_id']=id
        return vysledek.iloc[1:,:].set_index(['zdroj_id','id']).sort_index()

# bb=spocti_korelaci(aa,1000)
# b=spocti_korelaci(a.iloc[:,:2003],3000)

# srovnani=pd.concat([b,bb.rename(columns={'korelace':'korelace2'})],axis=1)

def spocti_korelace_pro_vsechny_uzivatele():
    ii=g[g.rating>1].index
    pocet=len(ii)
    i=0
    l=[]
    for id in ii:
        print("procesing user",i,"/",pocet)
        a=priprav_matici(id)
        b=spocti_korelaci(a,1000)
        if ~b.empty:
            l.append(b)
        i=i+1
    return l
        
l=spocti_korelace_pro_vsechny_uzivatele()

vysledek=pd.concat(l,axis=0)
vysledek=vysledek[~vysledek.korelace.isna()]

vysledek.to_pickle("D:\\Stepa\\CV\\Datasentics\\users_korelace.pcl")


# a=ratings_train.loc[(16,),:]
# ratings_train.loc[(11676,),:].loc[a.index,:]


