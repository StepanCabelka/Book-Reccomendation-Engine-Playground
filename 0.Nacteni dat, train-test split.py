import pandas as pd
import numpy as np
from scipy import sparse

users=pd.read_csv("D:\\Stepa\\CV\\Datasentics\\BX-users.csv", sep=";",encoding = "ISO-8859-1").rename(columns={'User-ID':'id', 'Location':'location','Age':'age'}).set_index('id')
users['country']=users['location'].str.split(',').str[-1]
users['loc_2']=users['location'].str.split(',').str[-2]
users['loc_3']=users['location'].str.split(',').str[-3]
users['loc_4']=users['location'].str.split(',').str[-4]
g=users.groupby('country')[['country']].count().rename(columns={'country':'pocet'}).sort_values(['pocet'],ascending=False)
l_divne=g.loc[g['pocet']==1,:].index.to_list()
users.loc[(users.country.isin(l_divne)),'country']='NA'
users.loc[users.country=='','country']='NA'

books=pd.read_csv("D:\\Stepa\\CV\\Datasentics\\BX-Books.csv", sep=';', encoding = "ISO-8859-1",warn_bad_lines=True, error_bad_lines=False).set_index('ISBN')
books=books.rename(columns={'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'published'})[['author','title','published']]

ratings=pd.read_csv("D:\\Stepa\\CV\\Datasentics\\BX-Book-Ratings.csv", sep=';', warn_bad_lines=True, encoding = "ISO-8859-1",error_bad_lines=False).rename(columns={'User-ID':'id', 'Book-Rating':'rating'}).set_index(['id','ISBN'])

#konzistence
i_users=users.index.sort_values()
i_users2=ratings.index.get_level_values(0).unique().sort_values()
i_users_final=i_users.intersection(i_users2)

i_books=books.index.sort_values()
i_books2=ratings.index.get_level_values(1).unique().sort_values()
i_books_final=i_books.intersection(i_books2)

users_final=users.loc[i_users_final,['age','country','loc_3']].rename(columns={'loc_3':'town'}).sort_index()
books_final=books.loc[i_books_final,:].sort_index()
ratings_final=ratings.reset_index()
ratings_final=ratings_final.loc[(ratings_final.id.isin(i_users_final)) & (ratings_final.ISBN.isin(i_books_final)),:].set_index(['id','ISBN'])

'''
train test split: no cold start in the test set, i.e. all users in the test must be from the train
'''
N=0.1 #pravdepodobnost vyberu konkretniho uzivatele z tech, co maji aspon dve recenze
g=ratings_final.groupby('id')['rating'].count()
gg=g[g>1]
np.random.seed(1)
l=[]

for id in gg.index:
    if np.random.rand()<N or id==108243:
        df=ratings_final.loc[(id,),:]
        n=df.shape[0]
        a=np.random.randint(n)
        df2=df.iloc[:a,:]
        df2['id']=id
        df2=df2.reset_index()
        l.append(df2)

ratings_test=pd.concat(l,axis=0).set_index(['id','ISBN'])
i=ratings_final.index.difference(ratings_test.index)
ratings_train=ratings_final.loc[i,:]

users_test=users_final.loc[ratings_test.index.get_level_values(0).drop_duplicates(),:]
users_train=users_final.loc[ratings_train.index.get_level_values(0).drop_duplicates(),:]

books_test=books_final.loc[ratings_test.index.get_level_values(1).drop_duplicates(),:]
books_train=books_final.loc[ratings_train.index.get_level_values(1).drop_duplicates(),:]

'''
kontrola: jsou vsichni users from test in train?
'''
# users_test.index.isin(users_train.index).all()
# ratings_test.index.get_level_values(0).drop_duplicates().isin(users_train.index).all()
# OK

'''
kontrola: jsou vsechny books from test in train?
'''
mask=(books_test.index.isin(books_train.index))
books_test=books_test[mask]
ratings_test2=ratings_test.reset_index()

# (books_test.index.isin(books_train.index)).all()

ratings_test=ratings_test2[ratings_test2.ISBN.isin(books_test.index)].set_index(['id','ISBN'])
u=pd.Index(ratings_test2.id.unique())
users_test=users_test.loc[u,:]

print(ratings_test.index.get_level_values(1).isin(ratings_train.index.get_level_values(1)).all())
print(books_test.index.isin(books_train.index).all())



'''
Ulozeni vycistenych dat
'''
users_test.to_pickle("D:\\Stepa\\CV\\Datasentics\\users_test.pcl")
books_test.to_pickle("D:\\Stepa\\CV\\Datasentics\\books_test.pcl")
ratings_test.to_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_test.pcl")


users_train.to_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl")
books_train.to_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl")
ratings_train.to_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl")

users_final.to_pickle("D:\\Stepa\\CV\\Datasentics\\users_final.pcl")
books_final.to_pickle("D:\\Stepa\\CV\\Datasentics\\books_final.pcl")
ratings_final.to_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_final.pcl")


'''
najdeme jednoho usera s vyhranenym vkusem, co cte harry pottera
'''

# a=books_final.loc[books_final.title.str.find('Potter')>-1,['author','title']]
# b=books_final.loc[books_final.author.isin(['J.K. Rowling','J. K. Rowling']),['author','title']]

# harry_potter_books=b.index

# users_harry_potter=ratings_final.reset_index()
# users_harry_potter=users_harry_potter.loc[users_harry_potter.ISBN.isin(harry_potter_books),'id'].unique()

# all_their_ratings=ratings_final.reset_index()
# all_their_ratings=all_their_ratings.loc[all_their_ratings.id.isin(users_harry_potter),['id','ISBN','rating']]

# g=all_their_ratings[all_their_ratings.rating>0].groupby('id')['id'].count().sort_values(ascending=False)
# g[300:400]

# r=ratings_final.loc[(108243,),:]
# books_final.loc[r.index,'title'].values
# ratings_test.loc[(108243,),:]

def user_books_rated(id):
    r=ratings_final.loc[(108243,),'rating']
    b=books_final.loc[r.index,['title']]
    r=pd.concat([r,b],axis=1)
    return r

# users_test.loc[108243,:]
# users_train.loc[108243,:]

# user_books_rated(108243)
    
# ratings_test.loc[(108243,),:]
# ratings_train.loc[(108243,),:]
