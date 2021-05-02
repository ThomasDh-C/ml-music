# ML Music
## By Thomas Dhome-Casanova and Morgan Teman
An investigation into creating the ultimate Taylor Swift song using machine learning
## Files and Folders:
- investigations - trying out different apis, packages and existing scripts
- data_collection - scraping scripts to collect all data needed and csvs of data collected
- lyric_generation - creating lyrics
- data_analysis - creating sound
- ./.env file (not tracked) - contains API keys (YOUTUBE_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, GENIUS_ID)
- ./RunMe - 'chmod +x ./RunMe' then './RunMe' to run all programs that generate lyrics
- ./InstallPackages - 'chmod +x ./InstallPackages' then './InstallPackages' to install packages used in this project
## Packages needed:
To install all packages run 'chmod +x ./InstallPackages' followed by './InstallPackages'

The following packages are used:
- pandas for storing dataframes
- selenium for web scraping
- beautiful soup (ended up not using this but found in investigations folder)
- tqdm for timing visualisations
- python-igraph for graph model (implemented in C++ so a lot faster than networkX)
- contractions for expanding contractions using regular expressions
- numpy 
- spacy and gensim for LDA analysis
- nltk for word analysis
- pqdict for a Min Priority queue
- Scipy for fitting normal distributions to data
- Sklearn for initial NMF and LDA analysis
- spotipy for Spotify API interactions
- openai for GPT3
