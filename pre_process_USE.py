#importing necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#importing movie sysnopys & review text
movie_details=pd.read_json("/kaggle/input/imdb-spoiler-dataset/IMDB_movie_details.json", lines=True)
imbd_reviews=pd.read_json("/kaggle/input/imdb-spoiler-dataset/IMDB_reviews.json", lines=True)


movie_details.head(4)
movie_details.rename(columns={'rating': 'movie_rating'}, inplace=True)

imbd_reviews.head(4)
imbd_reviews.rename(columns={'rating': 'review_rating'}, inplace=True)

#inner join of both dataframes
current_datetime = datetime.now()
big_df=pd.merge(movie_details,imbd_reviews,on='movie_id',how='inner')
current_datetime2 = datetime.now()
print(current_datetime2-current_datetime)


big_df=big_df.drop(['duration','genre','review_summary','user_id','movie_id'],axis=1)



#helper function to binarize Target
def binarize_target(x):
    lb = preprocessing.LabelBinarizer()
    lb.fit(['True','False'])
    return lb.transform(x)


#parameters for USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)


#helper function
def embed(input):
  return model(input)


#this uses helper function to input dataframe text column and output latent representaion of text using Universal Sentence Encoder
def preprocess(df,coln,start=0,end=None):
    arr=[]
    for i in tqdm(range(start,end)):
        arr.append(embed([df[coln][i]]))
    return np.array(arr)


#encoding & saving review text
review_text=np.squeeze(preprocess(big_df,"review_text",0,big_df.shape[0]))
# del review_text
np.save("review_txt.npy",review_text)

#encoding sysnopys text
synopsis_text=np.squeeze(preprocess(big_df,"plot_synopsis",0,big_df.shape[0]))

np.save("plot_synopsis.npy",synopsis_text)
