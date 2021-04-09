#%% Use beautiful soup (failed)
import requests
from bs4 import BeautifulSoup

# protected against beautiful soup
# code based on https://realpython.com/beautiful-soup-web-scraper-python/
URL = 'https://www.youtube.com/c/TaylorSwift/playlists?view=71&shelf_id=6'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find_all('a')
