import pandas as pd
import numpy as np
from collections import defaultdict

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


def precision_and_recall_at_k(both,k,threshold):
    user_est_true = defaultdict(list)
    for i,uid, _, true_r, est in both.itertuples():
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])
    
            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    
            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.
    
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return sum(prec for prec in precisions.values()) / len(precisions), sum(rec for rec in recalls.values()) / len(recalls)

def MAP(both,N,threshold):
    l_precision=[]
    l_recall=[0]
    for k in range(1,N+1):
        p,r=precision_and_recall_at_k(both, k, threshold)
        l_precision.append(p)
        l_recall.append(r)
    ch_recall=[l_recall[i]-l_recall[i-1] for i in range(1,N+1)]
    suma=0
    for k in range(N):
        suma=suma+l_precision[k]*ch_recall[k]
    return suma

df_books_books=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books-books collaborative recco predikce.pcl")
df_user_user=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\user-user collaborative recco predikce.pcl")
df_baseline=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\baseline recco predikce.pcl")
df_ranking=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\rating matrix recco predikce.pcl")
df_lightfm2_500=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf bookf predikce 500.pcl")
df_lightfm2_800=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf bookf predikce 800.pcl")
df_lightfm_500=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\Lightfm userf predikce 500.pcl")



for df,nazev in zip([df_baseline,df_books_books, df_user_user,df_ranking,df_lightfm_500,df_lightfm2_500,df_lightfm2_800],
                    ["baseline","book-book","user-user",'ranking','lightfm user feat.500','lightfm both feat.500','lightfm both feat.800']):
    both=pd.concat([ratings_test, df],axis=1).reset_index()
    both=both.rename(columns={'id':'uid','ISBN':'iid','rating':'true_r','predrating':'est'})
    p,r=precision_and_recall_at_k(both, 10, 4.9)
    m=MAP(both,10,4.9)
    print("*****",nazev,"*****")1761
    print(p,r,m)
    
    

    
    
    


    