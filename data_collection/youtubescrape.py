#%% Use selenium (a fake browser to scrape all data)
# import and set up general stuff
from selenium import webdriver
import time
import pandas as pd
from random import random
driver = webdriver.Chrome()



#%% Get all the playlists
URL = 'https://www.youtube.com/c/TaylorSwift/playlists?view=71&shelf_id=6'
driver.get(URL)

# scroll to page bottom https://stackoverflow.com/questions/32391303/how-to-scroll-to-the-end-of-the-page-using-selenium-in-python
def scroll_to_bottom(driver):
    old_position = 0
    new_position = None

    while new_position != old_position:
        # Get old scroll position
        old_position = driver.execute_script(
                ("return (window.pageYOffset !== undefined) ?"
                 " window.pageYOffset : (document.documentElement ||"
                 " document.body.parentNode || document.body);"))
        # Sleep and Scroll
        time.sleep(1)
        driver.execute_script((
                "var scrollingElement = (document.scrollingElement ||"
                " document.body);scrollingElement.scrollTop ="
                " scrollingElement.scrollHeight;"))
        # Get new position
        new_position = driver.execute_script(
                ("return (window.pageYOffset !== undefined) ?"
                 " window.pageYOffset : (document.documentElement ||"
                 " document.body.parentNode || document.body);"))

scroll_to_bottom(driver)
            
web_elem_list_playlists = driver.find_elements_by_id('video-title')
playlists = []
for playlist in web_elem_list_playlists:
    title = playlist.text
    url = playlist.get_attribute("href")

    new_playlist = {}
    new_playlist['title']= title
    new_playlist['url'] = url
    playlists.append(new_playlist)

# Output all the playlists and their urls ... will 
df = pd.DataFrame(playlists)
df.to_csv('data/youtubeplaylists_original.csv')



#%% Get the edited playlists (some are not on spotify so can't get key, some are extended cuts)
df = pd.read_csv('data/youtubeplaylists_edited.csv')
df = df.sample(frac=1).reset_index(drop=True) # shuffle to run away from any errors
output = []
driver.implicitly_wait(3) # seconds

urls = df['url']
playlists = list(df['title'])
youtubelink = 'https://www.youtube.com'
allsongobjs = []
for playlist_index, url in enumerate(urls):
    driver.get(url)
    time.sleep(random())
    box = driver.find_element_by_tag_name('ytd-playlist-panel-renderer')
    web_elem_list_songs = box.find_elements_by_id('playlist-items')

    for song_index, song in enumerate(web_elem_list_songs):
        text = song.find_element_by_id('video-title').text
        href = song.find_element_by_id('wc-endpoint').get_attribute("href")
        song_url = href
        obj = {'songname': text, 'songurl':song_url, 'playlist':playlists[playlist_index]}
        allsongobjs.append(obj)

finalsongobjs = []
for songobj in allsongobjs:
    url = songobj['songurl']
    driver.get(url)

    count = driver.find_element_by_id('info-contents').find_element_by_id('count')
    viewcount = count.find_elements_by_tag_name('span')[0].text
    newobj = songobj
    newobj['viewcount'] = viewcount
    finalsongobjs.append(newobj)

df = pd.DataFrame(finalsongobjs)
df.to_csv('data/song_stats.csv')

# %%
