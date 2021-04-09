#%% Scrape all data from various taylor swift youtube vids and store all jsons
# YOUTUBE API doesn't give any details that you can't get from webscraping
#%% Step 1: Load environment variables (no api keys on internet even if private) 
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')  # take environment variables from .env
YOUTUBE_KEY = os.getenv('YOUTUBE_KEY')

#%% Step 2: Let's find the channel
# code based on https://www.analyticssteps.com/blogs/how-extract-analyze-youtube-data-using-youtube-api
from googleapiclient.discovery import build
import pandas as pd

youtube = build('youtube','v3',developerKey=YOUTUBE_KEY)

# get the channelid
channelresponse = youtube.channels().list(part='contentDetails',forUsername='TaylorSwift').execute()
channelid = channelresponse['items'][0]['id'] # UCqECaJ8Gagnn7YCbPEzWH6g
uploadid = channelresponse['items'][0]['contentDetails']['relatedPlaylists']['uploads']
#check it is the right one
stats = youtube.channels().list(part='statistics',id=channelid).execute()['items'][0]['statistics']
subcount = int(stats['subscriberCount'])
if(subcount>40000000): print('found right channel')
else: print('found wrong channel')

# content = youtube.channels().list(part='contentDetails',id=channelid).execute()['items']
# hey = youtube.channels().list(part='contentDetails',id='UCqECaJ8Gagnn7YCbPEzWH6g').execute()
# hey = youtube.search().list(part='snippet', type='channel', q='Taylor Swift').execute()

# %% Get all playlists
# res = youtube.playlists().list(channelId=uploadid, part='snippet').execute()
res = youtube.playlists().list(part="snippet", channelId=channelid).execute()
playlistids = list( map(lambda x:x['id'], res['items']) )
playlisttitles = list( map(lambda x:x['snippet']['title'], res['items']) )
# only gives playlists she made, not albums
print(playlisttitles)

# %% Get all videos
allVideos = []
next_page_token = None
res = youtube.playlistItems().list(playlistId=uploadid, maxResults=50, part='snippet', pageToken = next_page_token).execute()

while True:
    res = youtube.playlistItems().list(playlistId=uploadid, maxResults=50, part='snippet', pageToken = next_page_token).execute()
    allVideos += res['items']
    next_page_token = res.get('nextPageToken')

    if(next_page_token == None): break

# summary stats
print('number of vids is :', len(allVideos))
playlistids = set( map(lambda x:x['snippet']['playlistId'], allVideos) )
# %%
