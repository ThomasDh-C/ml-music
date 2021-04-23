# %% --- Import libraries and json file -----------------------------
import json
import matplotlib.pyplot as plt
import re
plt.style.use('ggplot')


json_file = open('../data_collection/data/song_lyrics.json')
song_lyrics_list = json.load(json_file)

# %% --- Create dataframe of song data ------------------------------
import pandas as pd

# construct dataframe with 1 row per song of:
# song  .... its name
# verse_lyrics .... self explanatory
# bridge_lyrics, chorus_lyrics, outro_lyrics
# song order .... Verse 1, Verse 2, Chorus, Verse 3 etc
processed_song_list = []
song_obj = song_lyrics_list[0]
for song_obj in song_lyrics_list:
    if song_obj['lyrics'] == '': continue
    obj = {}
    obj['song'] = song_obj['song']

    text = song_obj['lyrics']
    # remove initial \n
    while text[0] == '\n':
        text = text[1:]
    verse_markers = re.findall(r'\[(\w+[ \w]*)\]', text)
    obj['song_order'] = verse_markers
    # ', '.join(verse_markers) if you wanna make this a string

    # split lyrics by the markers we just found .. regex also includes []
    obj['verse_lyrics'] = ''
    obj['bridge_lyrics'] = ''
    obj['chorus_lyrics'] = ''
    obj['outro_lyrics'] = ''
    obj['all_lyrics'] = ''

    # https://stackoverflow.com/questions/10974932/split-string-based-on-a-regular-expression
    lyrics_arr = re.split(r'\[\w+[ \w]*\]', text)[1:] # start at 1 as index 0 is ''
    for verse_name, lyric_segment in zip(verse_markers, lyrics_arr):
        # clean up starts of lyrics
        while lyric_segment[0].isalpha() != True:
            lyric_segment = lyric_segment[1:]
        
        # treat verses
        if(verse_name[:5] ==  'Verse'):
            obj['verse_lyrics'] = obj['verse_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Bridge'):
            obj['bridge_lyrics'] = obj['bridge_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Chorus'):
            obj['chorus_lyrics'] = obj['chorus_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Outro'):
            obj['outro_lyrics'] = obj['outro_lyrics'] + ' ' + lyric_segment
        
        # dump all lyrics with ' | ' between the verses
        obj['all_lyrics'] = obj['all_lyrics'] + ' | ' + lyric_segment

    # append to list
    processed_song_list.append(obj)


processed_song_df = pd.DataFrame(processed_song_list)


# %% --- Fix up the lyrics ------------------------------------------
from nltk.corpus import stopwords

orig_lyrics_list = processed_song_df['all_lyrics'].to_list()

# fix | and \n and ,  
orig_lyrics_list = [i.replace('|',' ') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace('\n',' ') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace(',',' ') for i in orig_lyrics_list]

# try to remove stopwords (clearer what is going on)
lyrics_list = [i.split(' ') for i in orig_lyrics_list]
english_stopwords = stopwords.words('english')
for c, lyrics in enumerate(lyrics_list):
    lyrics_list[c] = [i for i in lyrics if (i!='' and i not in english_stopwords)]
    orig_lyrics_list[c] = ' '.join(lyrics_list[c]).lower()

# fix - and '
orig_lyrics_list = [i.replace('-','') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace('-','') for i in orig_lyrics_list]




# %% --- NMF --------------------------------------------------------
# NMF and LDA based on following:
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

categories = 5
no_top_words = 20

# Display function
def display_topics(model, feature_names, no_top_words):
    for i, topic in enumerate(model.components_):
        print("Topic %d:" % i)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# NMF
# tf-idf represenation
tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)[\w']*") # token pattern is because we include ' in our tokens
tfidf = tfidf_vectorizer.fit_transform(orig_lyrics_list)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

nmf = NMF(n_components=categories, random_state=1, alpha=.1, l1_ratio=.4, init='nndsvd')
nmf.fit(tfidf)
# print the topics
display_topics(nmf, tfidf_feature_names, no_top_words)


# %% --- LDA --------------------------------------------------------
# LDA 
# bag of words
tf_vectorizer = CountVectorizer(token_pattern="(?u)[\w']*")
tf = tf_vectorizer.fit_transform(orig_lyrics_list)
tf_feature_names = tf_vectorizer.get_feature_names()
# create lda
# learning decay (0.5, 1.0] ... default .7
# learning_offset - greater than 1
lda = LatentDirichletAllocation(n_components=categories, learning_method='online', 
                                learning_offset=50 , learning_decay=.9, random_state=0, batch_size=10, max_iter=30)
lda.fit(tf)
print(lda.bound_)

# print the topics
print('------------------------------------------')
display_topics(lda, tf_feature_names, no_top_words)



# %% try gensim version in future
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0