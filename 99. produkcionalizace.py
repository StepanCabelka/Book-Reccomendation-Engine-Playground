import pandas as pd
from joblib import dump, load

import flask
import platform
platform.python_version()


'''
https://www.analyticsvidhya.com/blog/2020/04/how-to-deploy-machine-learning-model-flask/
https://medium.com/@dmahugh_70618/deploying-a-flask-app-to-google-app-engine-faa883b5ffab
'''

'''
****************************************************************************************************************
prerequisity
****************************************************************************************************************
'''
vysledekb=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_korelace.pcl")
avg_rating=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\average_rating.pcl")
books_final=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_final.pcl")

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

spocti_other_liked_books('B00011SOXI')

dump(spocti_other_liked_books,'D:\\Temp\\spocti_other_liked_books.txt')
# dump(vysledekb,'D:\\Temp\\vysledekb.txt')
# dump(avg_rating,'D:\\Temp\\avg_rating.txt')
# dump(books_final,'D:\\Temp\\books_final.txt')

'''
nacteni dulezitych objektu
'''

spocti_other_liked_books=load('D:\\Temp\\spocti_other_liked_books.txt')
vysedekb=load('D:\\Temp\\vysledekb.txt')
avg_rating=load('D:\\Temp\\avg_rating.txt')
books_final=load('D:\\Temp\\books_final.txt')

from flask import Flask, render_template, request, redirect, url_for


# function to get results for a particular text query
def requestResults(name):
    # get the tweets text
    tweets = get_related_tweets(name)
    # get the prediction
    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    # get the value counts of different labels predicted
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return data + str(tweets)


# start flask
app = Flask(__name__)
# render default webpage
@app.route('/')
def home():
    return render_template('D:\Stepa\Python\Datasentics\home.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

# get the data for the requested query
@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "

app.run(debug=True)













# class Model_doporuceni():
    
#     def __init__(self):
#         self.vysledek = pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\users_corelace.pcl")
#         self.avg_ratinbg = pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\average_rating.pcl")
#         self.ratings_train = ratings_train=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\ratings_train.pcl").sort_index()
#         # self.id = id
#         # self.how_many=how_many
    
#     # def train(self):
#     #     pass
        
    
#     def predict(self,id,how_many):
#         self.id = id
#         self.how_many=how_many
#         korelace=self.vysledek.loc[(self.id,),:]
#         suma=(korelace.sum()+0.001).values
#         relevant_ids=korelace.index.sort_values()
#         r=self.ratings_train.reset_index()
#         relevant_ratings=r.reset_index().loc[(r.id.isin(relevant_ids)) & (r.rating>0),:].set_index(['id','ISBN']).drop('index',1)
#         relevant_books=relevant_ratings.index.get_level_values(1).unique().sort_values()
#         matrix=np.zeros((len(relevant_ids),len(relevant_books)),dtype='int16')
#         for i in range(len(relevant_ids)):
#             for b in range(len(relevant_books)):
#                 t=(relevant_ids[i],relevant_books[b])
#                 if t in (relevant_ratings.index): 
#                     matrix[i,b]=relevant_ratings.loc[t,'rating']
#         matrix2=np.multiply(matrix,korelace.values)
#         dopo=matrix2.sum(axis=0)
#         dopo=dopo*1/suma
#         dopo_dict={relevant_books[i]:dopo[i] for i in range(dopo.shape[0])}
#         dopo_dict = {k: v for k, v in sorted(dopo_dict.items(), key=lambda item: item[1], reverse=True)}
#         df=pd.DataFrame(index=dopo_dict.keys(),data=dopo_dict.values(),columns=['rating'])
#         df.index.name='ISBN'
#         df=df[df.rating>0]
#         df=df.iloc[:self.how_many,:]
#         if df.shape[0]<self.how_many:
#             avg_r=avg_rating[~avg_rating.index.isin(df.index)]*0.5
#             df=pd.concat([df,avg_r],axis=0).iloc[:self.how_many,:].sort_values('rating',ascending=False)
#         return df        
    
#     # def predict(self):
#     #     return df

# m=Model_doporuceni()
# a=m.predict(8,28)        

# p=Pipeline(steps=[('nacti', Model_doporuceni())])    

# p.predict()    



# # dump the pipeline model
# dump(pipeline, filename="text_classification.joblib")


# # define the stages of the pipeline
# pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
#                                                       max_features=1000,
#                                                       stop_words= ENGLISH_STOP_WORDS)),
#                             ('model', LogisticRegression())])

# # fit the pipeline model with the training data                            
# pipeline.fit(train.tweet, train.label)


# # sample tweet
# text = ["Virat Kohli, AB de Villiers set to auction their 'Green Day' kits from 2016 IPL match to raise funds"]

# # predict the label using the pipeline
# pipeline.predict(text)

# from joblib import load

# # sample tweet text
# text = ["Virat Kohli, AB de Villiers set to auction their 'Green Day' kits from 2016 IPL match to raise funds"]

# # load the saved pipleine model
# pipeline = load("text_classification.joblib")

# # predict on the sample tweet text
# pipeline.predict(text)