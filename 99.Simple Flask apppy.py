import pandas as pd
from flask import request, Flask
app = Flask(__name__)
@app.route('/home')
def my_route():
  vysledekb=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_korelace.pcl")
  avg_rating=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\average_rating.pcl")
  books_final=pd.read_pickle("D:\\Stepa\\CV\\Datasentics\\books_final.pcl")
  isbn= request.args.get('isbn', default = 'NA', type = str)
  
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

  df=spocti_other_liked_books(isbn)
  df.index.name='isbn'
  df['zdroj_isbn']=isbn
  df=df.reset_index()[['zdroj_isbn','isbn','rating','typ','title']]
  return df.to_html(header="true", table_id="table",index=False)

'''
example
/my-route?isbn='B0002JV9PY'
'''
if __name__ == '__main__':
    app.run()





# @app.route('/')
# def hello():
#     if 'isbn' in request.args:
#         isbn = request.args['isbn']
#     else:
#         isbn = ''
#      return HELLO_HTML.format(
#              name, str(datetime.now()))

#  HELLO_HTML = """
#      <html><body>
#          <h1>Hello, {0}!</h1>
#          The time is {1}.
#      </body></html>"""

#  if __name__ == "__main__":
#      # Launch the Flask dev server
#      app.run(host="localhost", debug=True)


# import pandas as pd
# data={'ISBN':['a','b','c'],
#       'title':['Harry potter', 'Wuthering Heights', 'Study in Scarlet'],
#       'rating':[7.8, 5.6, 1.4]}

# df=pd.DataFrame(data=data)

# from flask import Flask
# app = Flask(__name__)
# @app.route('/')
# def hello_world():
#     return df.to_html(header="true", table_id="table")

# if __name__ == '__main__':
#     app.run()
    