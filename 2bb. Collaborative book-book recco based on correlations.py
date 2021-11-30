import pandas as pd
import numpy as np

vysledekb=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_korelace.pcl")
books_vysledekb=vysledekb.index.get_level_values(0).drop_duplicates()
vysledekb2=vysledekb.reset_index()
vysledekb2_positive=vysledekb2[vysledekb2.korelace>0]
vysledekb_positive=vysledekb[vysledekb.korelace>0]
isbn_s_cor=vysledekb.index.get_level_values(1).drop_duplicates()


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

id=56959
# ratings_train2.loc[ratings_train2.ISBN=='B00011SOXI',:]

def spocti_doporuceni_positive_cor(id,how_many):
    his_ratings=ratings_train_positive.loc[(id,),:].sort_index()
    his_books=his_ratings.index
    correlated_books=vysledekb2_positive.loc[vysledekb2_positive.ISBN_zdroj.isin(his_books),:]
    if ~correlated_books.empty:
        matrix=correlated_books.pivot(index='ISBN_zdroj',columns='ISBN',values='korelace').sort_index()
        matrix_sum=matrix.sum(axis=0)
        soucin=np.repeat(his_ratings.loc[matrix.index,:].values.reshape(matrix.shape[0],1),matrix.shape[1]).reshape(matrix.shape)*(matrix)
        soucin_sum=soucin.sum(axis=0)
        df=pd.DataFrame(soucin_sum/matrix_sum).rename(columns={0:'rating'}).sort_values('rating',ascending=False)
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

# a=spocti_doporuceni_positive_cor(56959, 200)

id=56959
isbn='0425117383'


def spocti_rating(id,isbn):
    global_dop=avg_rating.loc[isbn,'rating']
    if isbn in isbn_s_cor:
        this_book_correlations=vysledekb.loc[(slice(None),isbn),:]
        this_book_correlations=this_book_correlations[this_book_correlations.korelace>0]
        if not(this_book_correlations.empty):
            his_ratings=ratings_train.loc[(id,),:].sort_index()
            his_ratings=his_ratings[his_ratings.rating>0]
            his_books=his_ratings.index
            correlated_books=this_book_correlations.loc[(his_books,),:]
            his_ratings=his_ratings.loc[correlated_books.index.get_level_values(0),:]
            if not(his_ratings.empty):
                # print(his_ratings)
                # print(~his_ratings.empty)
                rating=(correlated_books.values*his_ratings.values).sum()/correlated_books.values.sum()
                return rating
            else:
                return global_dop
        else:
            return global_dop
    else:
        return global_dop

# spocti_rating(199,'0373196407')

def spocti_predikci():
    df=pd.DataFrame(index=ratings_test.index,columns=['predrating'])
    counter=0
    for id,isbn in df.index:
        counter=counter+1
        print('computing',counter,'of',df.shape[0])
        df.loc[(id,isbn),'predrating']=spocti_rating(id,isbn)
    return df

# isbn='B00011SOXI'


def spocti_other_liked_books(isbn,how_many=20):
    '''
    predpokladam: ze dotycna kniha ma hodnoceni 10
    '''
    if isbn in vysledekb.index.get_level_values(0).drop_duplicates():
        isbn_korelovane=vysledekb.loc[(isbn,),:].sort_index()
        avg_rating_korelovanych=avg_rating.loc[isbn_korelovane.index,'rating']
        both=pd.concat([isbn_korelovane,avg_rating_korelovanych],axis=1)
        both.loc[both.korelace>0,'final_rating']=both.rating*both.korelace
        both.loc[both.korelace<=0,'final_rating']=(1-both.rating)*both.korelace
        df=both[['final_rating']].rename(columns={'final_rating':'rating'})
        df['typ']='korelace'
        df=df.sort_values('rating',ascending=False).iloc[:how_many,:]
    else:
        df=pd.DataFrame(columns=['rating'])
    if df.shape[0]<how_many:
        avg_r=avg_rating[~avg_rating.index.isin(df.index)]*0.5
        avg_r=avg_r.iloc[:how_many-df.shape[0],:]
        avg_r['typ']='global'
        df=pd.concat([df,avg_r],axis=0).iloc[:how_many,:].sort_values('rating',ascending=False)
        df['title']=books_final.loc[df.index,'title']
    return df

# spocti_other_liked_books('B00011SOXI')
    
def spocti_all_liked_books():
    reco=[]
    counter=0
    pocet=len(books_final.index)
    for isbn in books_final.index:
        counter=counter+1
        # if counter%1000==0:
        print("counter",counter,"of",pocet)
        reco.append(spocti_other_liked_books(isbn))
    return reco

all_liked_books=spocti_all_liked_books()

    










df=spocti_predikci()

df.to_pickle("D:\\Stepa\\CV\\Datasentics\\books-books collaborative recco predikce.pcl")


'''
metriky pro mereni uspesnosti: precission at k and recall at k
'''
'''precision at k'''

both=pd.concat([ratings_test, df],axis=1).reset_index().rename(columns={'id':'uid','ISBN':'iid','rating':'true_r','predrating':'est'})

p,r=precision_and_recall_at_k(both, 10, 5)





    
    
    


    