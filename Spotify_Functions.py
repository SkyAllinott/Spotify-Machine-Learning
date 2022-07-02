import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# This file contains all the functions used in spotify_api.py and data_analysis.py
with open('IDs.txt') as read:
    lines = read.readlines()
cid = lines[0]
secret = lines[1]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret))

def showTracks(results, uriArray):
    for i, item in enumerate(results['items']):
        track = item['track']
        uriArray.append(track['id'])

def playlistTrackId(username, playlist_id):
    trackId=[]
    results = sp.user_playlist(username, playlist_id)
    tracks = results['tracks']
    showTracks(tracks, trackId)
    while tracks['next']:
        tracks=sp.next(tracks)
        showTracks(tracks, trackId)
    return trackId

def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)

    # meta
    name = meta['name']
    track_id = meta['uri']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    artist_id = meta['artists'][0]['uri']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    type = meta['type']
    explicit = meta['explicit']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    time_signature = features[0]['time_signature']
    valence = features[0]['valence']
    key = features[0]['key']

    track = [name, track_id, album, artist, artist_id, release_date, length, popularity, type, explicit, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature, valence, key]
    return track

def graph(x, y, data, yunit=""):
    """
    Scatter plot of x and y, with trend line and updated axis labels
    :param x: x data
    :param y: y data
    :param data: pandas dataframe
    :param yunit: Provide optional yunits as a string
    :return:
    """
    fig = plt.figure()
    z = np.polyfit(data[x], data[y], 1)
    p = np.poly1d(z)
    plt.plot(data[x], p(data[x]), color="r")
    plt.scatter(data[x], data[y], color='black')
    title1 = str(["Relationship between " + y + " and " + x])
    len1 = len(title1)-2
    title1 = title1[2:len1]
    plt.title(title1)
    plt.xlabel(x.capitalize())
    if yunit == "":
        plt.ylabel(y.capitalize())
    else:
        plt.ylabel(y.capitalize() + " (" + yunit + ")")
    return fig

def histogram(x, data, nbins=""):
    """
    Plots histogram with updating title
    :param x: data to histogram
    :param data: pandas dataframe
    :param nbins: (optional) bins for histogram
    :return:
    """
    fig = plt.figure()
    title = str("Distribution of " + x)
    plt.title(title)
    if nbins == "":
        plt.hist(data[x])
    else:
        plt.hist(data[x], bins= nbins)
    title = str("Distribution of " + x)
    title
    plt.title(title)
    return fig