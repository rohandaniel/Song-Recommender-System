import numpy as np
import pandas as pd


from scipy.spatial.distance import cdist
from collections import defaultdict


import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import spotipy

data = pd.read_csv("C:/Users/Daniel/Desktop/Final Yr. Project/data/data.csv")

song_cluster_pipeline = pkl.load(open("song_recommender_system.pickle", "rb"))


from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "c90b0c01c00b4213925c9d38752a8359"
CLIENT_SECRET = "f6ec9781d6b54680a839307895f9ed65"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET))



#Finds song details from spotify dataset. If song is unavailable in dataset, it returns none.

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

#Fetches song details from dataset. If info is unavailable in dataset, it will search details from the spotify dataset.

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        print('Fetching song information from local dataset')
        return song_data
    
    except IndexError:
        print('Fetching song information from spotify dataset')
        return find_song(song['name'], song['year'])


#Fetches song info from dataset and does the mean of all numerical features of the song-data.

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))#nd-array where n is number of songs in list. It contains all numerical vals of songs in sep list.
    #print(f'song_matrix {song_matrix}')
    return np.mean(song_matrix, axis=0) # mean of each ele in list, returns 1-d array



#Flattenning the dictionary by grouping the key and forming a list of values for respective key.

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys(): 
        flattened_dict[key] = [] # 'name', 'year'
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value) # creating list of values
    return flattened_dict



#Gets song list as input. 
#Get mean vectors of numerical features of the input. 
#Scale the mean-input as well as dataset numerical features.
#calculate eculidean distance b/w mean-input and dataset.
#Fetch the top 10 songs with maximum similarity.

def recommend_songs( song_list, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, data)
    #print(song_center)
    #print(type(song_center))
    
    scaler = song_cluster_pipeline.steps[0][1] # StandardScalar()
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    #print(f'distances {distances}')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')
    

st.set_page_config(page_title="Song Recommender System",
                   #page_icon=mimiric_logo,
                   #layout='wide',
                   #initial_sidebar_state='auto'
                   )

col1, mid, col2 = st.columns([1,1,10])
with col1:
    st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYK1O9Y67h6Ae6U880tW9rxSYKc_dUENj72Q&usqp=CAU",
            width=120, # Manually Adjust the width of the image as per requirement
        )
with col2:
    st.header('Song Recommender System')


n = int(st.text_input('Input number of songs you are going to add here:'))

lst=[]
for x in range(n):
    name=st.text_input('Input name of the song:',key=x)
    year=int(st.text_input('Input the year when the song was released:',key=x+n))
    lst.append({'name': name, 'year':year})
number=st.text_input('Input number of song needed for the suggestion:')
if number:
    lst1=recommend_songs(lst,int(number))
else:
    lst1=recommend_songs(lst)
    

name=[]
year=[]
artists=[]
for x in lst1:
    name.append(x["name"])
    year.append(x["year"])
    artists.append((", ".join(x["artists"][2:-2].split("', '"))))
dict1={"Name":name,"Year":year,"Artists":artists}

st.write("Suggested Songs are:")
st.table(dict1)   
    
    