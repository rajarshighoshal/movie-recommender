import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

# data loading
print('Reading CSV...', end='')
metadata = pd.read_csv('movies_metadata.csv')
metadata = metadata.drop_duplicates(subset=['id'])
movie_dt = metadata[['genres', 'title', 'overview', 'tagline']]
print('...Done...')

# data cleaning
print('Cleaning Data...', end='')
from ast import literal_eval
movie_dt.genres = movie_dt.genres.fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# create feature vetors
movie_dt.tagline = movie_dt.tagline.fillna('')
movie_dt.overview = movie_dt.overview.fillna('')
movie_dt['content'] = movie_dt.tagline + movie_dt.overview
final_df = movie_dt[['genres', 'title', 'content']]
final_df['genres'] = final_df['genres'].apply(lambda x : ' '.join(x))
# remove na
final_df.dropna(inplace=True)
# remove duplicates
final_df = final_df.drop_duplicates(subset=['title'])
print('...Done...')

# create recommendation using cosine similarity
print('Building model...')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_content = tf.fit_transform(final_df['content'])
print('...still working...')
cosine_sim_content = linear_kernel(tfidf_content, tfidf_content)
print('...just few more minutes...')
count_vectorizer = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_genre = count_vectorizer.fit_transform(final_df['genres'])
print('...almost there...')
cosine_sim_genre = cosine_similarity(count_genre, count_genre)
print('...Done...')

# save cosine similairties
print('Saving model and data...')
import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('content', data=cosine_sim_content)
print('...almost there...')
h5f.create_dataset('genre', data=cosine_sim_genre)
h5f.close()
final_df.to_pickle('movie_titles.pkl')
print('...Done, model is ready, you can run recommender system now.')