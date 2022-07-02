import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import sys
sys.path.append('G:/My Drive/Python Projects/Spotify API')
import Spotify_Functions as sf


# Place access tokens inside your own .txt file:
with open('IDs.txt') as read:
    lines = read.readlines()
cid = lines[0]
secret = lines[1]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret))

# Returns trackIDs for all songs in playlist.
ids = sf.playlistTrackId('bretallinott', '0881GMlvqjXJIZxj5mjT44')

# Returns all the track features, given the id.
tracks = []
for i in range(len(ids)):
    track = sf.getTrackFeatures(ids[i])
    tracks.append(track)

# Dataframe conversion and some data clean up:
df = pd.DataFrame(tracks, columns = ['name', 'track_id', 'album', 'artist', 'artist_id', 'release_date', 'length', 'popularity', 'type', 'explicit', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature', 'valence', 'key'])
df['length'] = df['length']/60000

df['release_year'] = df['release_date'].str[:4]
df['release_month'] = df['release_date'].str[5:7]
df = df[df['release_month'] != ""]
df = df.astype({"release_year": int, "release_month":int })
df = df.drop('release_date', axis = 1)
df['explicit'] = df['explicit'].astype(int)


# Pulling artist information for genres, followers, and their unique artist id.
L = np.array(df['artist_id'])
lengthL = len(L)
results = np.zeros([lengthL, 1])
artstrings = np.zeros([lengthL, 2], dtype='object')
results = np.hstack([results, artstrings])
for i in range(len(L)):
    x = sp.artist(L[i])
    results[i, 0] = x['followers']['total']
    results[i, 1] = x['uri']
    results[i, 2] = x['genres']
results = pd.DataFrame(results, columns=['followers', 'artist_id', 'genres'])


# Merging the artist info on track info, on the artist id.
df = df.merge(results)
# many to many conversion leads to duplicates, so they are removed based on track_id, which is unique.
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True, ignore_index=True)

df['followers'] = df['followers'].astype(int)
df['genre'] = df['genres'].str[0]
df = df.drop('genres', axis = 1)

# Output to CSV for analysis in data_analysis.py
df.to_csv("spotify.csv", sep=',')
