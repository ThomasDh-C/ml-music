#%% Step 1: Load environment variables (no api keys on internet even if private) 
# add current directory as place to search for libraries
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../../.env')  # take environment variables from .env
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
GENIUS_ID = os.getenv('GENIUS_ID')

import sys
sys.path.append("./")

#%% Step 2: create object and get lyrics
from get_lyrics import GetLyrics 

lyrics_class = GetLyrics(spotify_client_id = SPOTIFY_CLIENT_ID, 
                        spotify_client_secret = SPOTIFY_CLIENT_SECRET, 
                        user_id = 'https://open.spotify.com/user/mteman515?si=2c32d4f2f22140c9', 
                        playlist_id = 'https://open.spotify.com/playlist/5KZWk2UlJFZMzp27HtKx4M?si=b6c309aea0c6454b', 
                        genius_key = GENIUS_ID)

lyrics_string = lyrics_class.get_lyrics()
track_names = lyrics_class.get_track_names()
artists = lyrics_class.get_track_artists()

# %% Step 3: Dump all that knowledge into a json file
import json

json_output = []
for i in range(len(track_names)):
    obj = {}
    obj['song'] = track_names[i]
    obj['artist'] = artists[i]
    obj['lyrics'] = lyrics_string[i]
    json_output.append(obj)

with open("../data/song_lyrics.json", 'w') as outfile:
    json.dump(json_output, outfile)
print('Data exported to ../data/song_lyrics.json')
# %%
