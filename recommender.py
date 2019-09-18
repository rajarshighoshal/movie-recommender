import h5py
import pandas as pd
import numpy as np



# create function for recommendation
def get_recommendations(title, n):
    """
    params :title movie title for which recommendation would be whown
           :n number of recomendation needed to be shown
    return :n number of recommended movies based on :title
    """
    # load data files

    print('Loading the database for you....')
    try:
        h5f = h5py.File('data.h5','r')
    except OSError:
        return 'Please Run dataLoder First!'
    content = h5f['content'][:]
    print('almost there.....')
    genre = h5f['genre'][:]
    print('just few more seconds...')
    h5f.close()
    final_df = pd.read_pickle('movie_titles.pkl')
    final_df = final_df.reset_index()
    titles = final_df['title']
    indices = pd.Series(final_df.index, index=final_df['title'])
    print('Done...\nRecommending your movies....')
    idx = indices[title]
    sim_scores = list(enumerate(content[idx])) + list(enumerate(genre[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    print('Your recommended movies are:-')
    return titles.iloc[movie_indices]


if __name__ == '__main__':
    movie = input('Enter a movie you have really liked: ')
    n = int(input('Enter number of movies you want to see: '))
    try:
        print(get_recommendations(movie, n))
    except KeyError :
        print("Opps! Sorry! The movie doesn't seem to exist in out Database.\nCheck your spelling and the name is case-sensitive; check that as well.\nBetter luck next time.")
    
