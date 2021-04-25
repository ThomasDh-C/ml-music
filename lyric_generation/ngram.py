from numpy.core.numeric import NaN
import pandas as pd
import contractions
import re
import numpy as np

processed_song_df = pd.read_csv('temp/processed_song_df.csv')
processed_song_df.drop('Unnamed: 0', axis=1, inplace=True)
lyrics = processed_song_df['all_lyrics']

# %% --- fix phrases up ---------------------------------------------------------
def phrase_fix(phrase):
    phrase = re.sub(r"\n\n \| ", " endofparagraph ", phrase) # end of paragraph
    phrase = re.sub(r"\n\n", " endofparagraph", phrase) # end of song
    phrase = re.sub(r"\n", " endofline ", phrase) # end of line

    phrase = re.sub(r",", "", phrase) # remove all commas
    phrase = re.sub(r"\?", "", phrase) # remove all ?
    phrase = re.sub(r"-|\.|\"|\(|\)", "", phrase) # remove all hyphens dots "
    
    # https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/
    # tried word2vec method but wayyy too slow with pycontractions but way too slow
    contractions.add('\'bout', 'about')
    phrase = contractions.fix(phrase)

    phrase = phrase.lower()

    return phrase

new_lyrics = []
for song in lyrics:
    if (type(song) == float): continue
    testphrase = song[3:]
    testphrase = phrase_fix(testphrase)
    new_lyrics.append(testphrase)


pd.DataFrame(new_lyrics).to_csv('temp/phrase_fix_1.csv')

# %% --- build up a bigram model -------
all_words = set()
all_words.add('')
# word: {next1: count, next2: count}
bigram_words_counts = {}
for song in new_lyrics:
    song_arr = [word for word in song.split(' ') if word!='']
    prev_word = ''
    for word in song_arr:
        if prev_word not in bigram_words_counts: bigram_words_counts[prev_word] = {}
        count = bigram_words_counts[prev_word].get(word,0)
        bigram_words_counts[prev_word][word] = count+1

        all_words.add(word)
        prev_word = word


# laplace smooth
for word, next_words_dict in bigram_words_counts.items():
    for nextword in all_words:
        bigram_words_counts[word][nextword] = next_words_dict.get(nextword,0) + .3




# %% --- run bigram model
prev_word=''
words = []
for i in range(100):
    # add in some randomness
    p_dist = bigram_words_counts.get(prev_word)
    for w in p_dist.keys():
        p_dist[w] = p_dist[w]+np.random.normal(0,1.5)
    # select a word
    new_word = max(p_dist, key=p_dist.get)
    # append word and iterate
    words.append(new_word)
    prev_word=new_word

outputstr = ' '.join(words)
outputstr = re.sub(r" endofparagraph ", " \n\n", outputstr) # end of song
outputstr = re.sub(r" endofline ", " \n", outputstr) # end of line
print(outputstr)
print('----------------------------------------------------------------')




# %% ---idea 2 on bigram model - shortest path 
# say we want a sentence of 7 words
# we have to get from start "" node to the 'endofline' in 7 moves
# so make a network with:
#  1 node layer ("") -> 2193 node layer (repeat 7 layers for 7word) --> 1 node layer 'end of line'
import networkx as nx
from kshortestpath import k_shortest_paths
from tqdm import tqdm

G = nx.Graph()

# add nodes
layersource = ['']
inbetweenlayers = []
for i in range(7):
    words = [str(i)+"_"+word for word in all_words]
    inbetweenlayers=inbetweenlayers+words
finallayer = ['endofline']
G.add_nodes_from(layersource+inbetweenlayers+finallayer)

# add weights as weight = 1/freq (find shortest path with non-neg weights)
# layersource to layer0
orig_word_dict = bigram_words_counts.get('')
for word in all_words:
    node_key = "0_"+word
    weight = 1/orig_word_dict[word]
    G.add_edge('', node_key, length=1/orig_word_dict[word])

# iterate through all the layers index 0,1,2, 3,4,5 (layer 6 is a closing layer)
for i in tqdm(range(6)):
    # iterate through all left nodes
    for left_word in all_words:
        orig_word_dict = bigram_words_counts.get(left_word)
        left_node_key = str(i) + "_" + left_word
        # iterate through all right nodes
        for right_word in all_words:
            right_node_key = str(i) + "_" + right_word
            weight = 1/orig_word_dict[right_word]
            if(weight<0): print((left_node_key, right_node_key))
            G.add_edge(left_node_key, right_node_key, length=weight)
        
# layer 6 is a closing layer
for left_word in all_words:
    orig_word_dict = bigram_words_counts.get(left_word)
    left_node_key = "5_" + left_word
    weight = 1/orig_word_dict['endofline']
    G.add_edge(left_node_key, 'endofline', length=weight)

# https://github.com/guilhermemm/k-shortest-path/blob/master/k_shortest_paths.py

output = k_shortest_paths(G=G, source='', target='endofline', k=1, weight="length")
print(output)









# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
all_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
ngram_words_counts = {}
for song in new_lyrics:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in ngram_words_counts: ngram_words_counts[prev_word] = {}
        count = ngram_words_counts[prev_word].get(word,0)
        ngram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        all_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    


# laplace smooth
for word, next_words_dict in ngram_words_counts.items():
    for nextword in all_words:
        count = next_words_dict.get(nextword,0)
        ngram_words_counts[word][nextword] = count + .3
unknown_dict = {word: .3 for word in all_words}

# %% --- run ngram model
from copy import deepcopy
prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(100):
    # add in some randomness
    p_dist = {}
    if prev_word in ngram_words_counts.keys():
        p_dist = ngram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(unknown_dict) # if unseen use unknown dict
    
    for w in p_dist.keys():
        p_dist[w] = p_dist.get(w)+np.random.normal(0,1)
    # select a word
    new_word = max(p_dist, key=p_dist.get)
    # append word and iterate
    words.append(new_word)
    prev_word_arr = prev_word_arr[1:] + [new_word]
    prev_word = " ".join(prev_word_arr)

outputstr = ' '.join(words)
outputstr = re.sub(r" endofparagraph ", " \n\n", outputstr) # end of song
outputstr = re.sub(r" endofline ", " \n", outputstr) # end of line
print(outputstr)
# %%
