import pandas as pd
import numpy as np
from scipy import sparse


'''
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?#h2_10
'''

'''
načtení training data 
'''
users_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_train.pcl").sort_index()
books_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_train.pcl").sort_index()
ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
ratings_train['rating']=ratings_train['rating'].astype('int8')

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


'''
zmenšení matice
'''

r=ratings_train[ratings_train.rating>0]

#dále dropnu usery a knihy, co maji malo ratingu
value=True
while(value):
    g=r.reset_index().groupby('id')[['rating']].count()
    zredukuj_users=g[g.rating==1].shape[0]
    g=g[g.rating>1]
    r=r.reset_index().loc[r.reset_index().id.isin(g.index),:]
    gg=r.groupby('ISBN')[['rating']].count()
    zredukuj_books=gg[gg.rating==1].shape[0]
    print(zredukuj_users,zredukuj_books)
    gg=gg[gg.rating>1]
    r=r.loc[r.ISBN.isin(gg.index),:].set_index(['id','ISBN'])
    if zredukuj_users+zredukuj_books==0:
        value=False

i_users=r.index.get_level_values(0).drop_duplicates().sort_values()
i_books=r.index.get_level_values(1).drop_duplicates().sort_values()
users=users_train.loc[users_train.index.isin(i_users),:].sort_index()
books=books_train.loc[books_train.index.isin(i_books),:].sort_index()

R=np.empty((users.shape[0],books.shape[0]),dtype='int8')
# R.nonzero()
# R.shape

user_dict={users.index[i]:i for i in range(users.shape[0])}
isbn_dict={books.index[i]:i for i in range(books.shape[0])}
samples=[]

for (id,isbn),ra in r.itertuples():
    i=user_dict[id]
    j=isbn_dict[isbn]
    R[i,j]=ra
    samples.append((i,j,ra))
# samples = [
#         (i, j, R[i, j])
#         for i in range(R.shape[0])
#         for j in range(R.shape[1])
#         if R[i, j] > 0
#         ]

del users_train, books_train, ratings_train
del g, gg 

class MF():
    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix 
    def train(self):
        print('initializing P and Q')
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K)).astype('float32')
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K)).astype('float32')

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        print('generating samples')
        # List of training samples
        self.samples = samples

        # Stochastic gradient descent for given number of iterations
        training_process = []
        print('running training process')
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            print("Iteration: %d ; error = %.4f" % (i+1, mse))
            training_process.append((i, mse))
        # if (i+1) % 20 == 0:
            

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        #predicted = self.full_sample_predicted()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[(x, y)], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_sample_predicted(self):
        d={}
        for i, j, r in self.samples:
            d[(i,j)]=self.get_rating(i, j)
        return d    
        
    def full_matrix(self):
        print("computing full matrix")
        A=self.P.dot(self.Q.T) 
        A=A+self.b
        for col in range(A.shape[1]):
            A[:,col]=A[:,col]+self.b_u

        for row in range(A.shape[0]):
            A[row,:]=A[row,:]+self.b_i
        return A
    

# R= np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))
# del A
np.random.seed(25)
mf = MF(R, K=500, alpha=0.001, beta=0.01, iterations=100)
mf_train=mf.train()

P=mf.P
Q=mf.Q
b=mf.b
b_u=mf.b_u
b_i=mf.b_i.reshape(1,-1)

A=P.dot(Q.T) 
A=A+b
for col in range(A.shape[1]):
    A[:,col]=A[:,col]+b_u

for row in range(A.shape[0]):
    A[row,:]=A[row,:]+b_i

output=pd.DataFrame(index=users.index, columns=books.index, data=A)
output.to_pickle("D:\\Stepa\\CV\\Datasentics\\Factorization k=500.pcl")

#K=1000, error 656.3137
#K=200, error 662.2539

def spocti_doporuceni(id,how_many):
    if (id in output.index):
        df=pd.DataFrame(output.loc[id,:]).rename(columns={id:'rating'}).sort_values('rating',ascending=False)
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

# a=spocti_doporuceni(278832,300000)




def spocti_rating(id,isbn):
    if (id in user_dict.keys()) & (isbn in isbn_dict.keys()):
        (i,j)=user_dict[id], isbn_dict[isbn]
        print('vracim matici')
        return A[i,j]
    else:
        print('vracim baseline')
        return avg_rating.loc[isbn,'rating']

# spocti_rating(8, '0060973129')

def spocti_predikci():
    df=pd.DataFrame(index=ratings_test.index,columns=['predrating'])
    counter=0
    for id,isbn in df.index:
        counter=counter+1
        print('computing',counter,'of',df.shape[0])
        df.loc[(id,isbn),'predrating']=spocti_rating(id,isbn)
    return df

df=spocti_predikci()
df.to_pickle("D:\\Stepa\\CV\\Datasentics\\rating matrix recco predikce.pcl")