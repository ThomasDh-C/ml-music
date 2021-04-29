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

# %% --- build up a bigram counts model -------
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

# apply prior


# %% --- shortest path ---------------------
# say we want a sentence of 7 words
# we have to get from start "" node to the 'endofline' in 7 moves
# so make a network with:
#  1 node layer ("") -> 2193 node word layer (repeat 7 layers for 7word) --> 1 node layer 'end of line'
import networkx as nx
from tqdm import tqdm
from itertools import islice

n_words = 7

G = nx.DiGraph()

# add nodes
layersource = ['']
inbetweenlayers = []
for i in range(n_words):
    words = [str(i)+"_" + word for word in all_words]
    inbetweenlayers = inbetweenlayers + words
finallayer = ['endofline']
G.add_nodes_from(layersource+inbetweenlayers+finallayer)

# add weights as weight = 1/freq (find shortest path with non-neg weights)
# layersource to layer0
orig_word_dict = bigram_words_counts.get('')
for word in all_words:
    node_key = "0_" + word
    weight = 1/orig_word_dict[word]
    G.add_edge('', node_key, length=1/orig_word_dict[word])
    if(weight<0): print(('', node_key)) # error checking
    
no_out_connections = {'', 'endofline', 'endofparagraph'}
# iterate through all the layers index 0->1->2->3->4->5-> (layer 6 is a closing layer)
for i in tqdm(range(n_words-1)):
    # iterate through all left nodes
    for left_word in all_words:
        # skip making out connections from these words
        if(left_word in no_out_connections): continue

        orig_word_dict = bigram_words_counts.get(left_word)
        left_node_key = str(i) + "_" + left_word
        # iterate through all right nodes
        for right_word in all_words:
            right_node_key = str(i+1) + "_" + right_word
            weight = 1/orig_word_dict[right_word]
            if(weight<0): print((left_node_key, right_node_key)) # error checking

            G.add_edge(left_node_key, right_node_key, length=weight)
        
# layer 6 is a closing layer
for left_word in all_words:
    orig_word_dict = bigram_words_counts.get(left_word)
    left_node_key = str(n_words-1) + "_" + left_word
    weight = 1/orig_word_dict['endofline']
    if(weight<0): print((left_node_key, 'endofline')) # error checking
    G.add_edge(left_node_key, 'endofline', length=weight)

print('graph made')


# https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

print('working on sentences')
for path in k_shortest_paths(G, source='', target='endofline', k=10, weight="length"):
    # path is an array of nodes passes through --> convert to sentence
    path = path[1:][:-1]
    out_arr = [element[2:] for element in path]
    out_arr[0] = out_arr[0].capitalize()
    print(" ".join(out_arr))

# currently prints
# I do not you know that you
# I do not know i want you
# I do not know i knew you
# I do not you do not you
# I do not know i would you
# I know that i do not you
# I can not you know that you
# I do not you are not you
# I can not know i want you
# I can not know i knew you

# %% --- Next steps ---------------------------------------------------------
# https://stackoverflow.com/questions/34909588/how-to-remove-random-number-of-nodes-from-a-network-graph
# remove random nodes and get the best sentences
# choose the best one of the top 10 for reasoning (somehow do coherence magic?)
# generate tons of lines with this ... or maybe lines for chorus, lines for whatever etc