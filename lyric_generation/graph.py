from numpy.core.numeric import NaN
import pandas as pd
import contractions
import re
import numpy as np
import json

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
json_file = open('./temp/lda_words_categ1.json')
prior_input = json.load(json_file) # list of ['word', probability_float]
prior_word_set = set([member[0] for member in prior_input])
matching_words_set = prior_word_set&all_words
print('applying prior of ' + str(len(matching_words_set)) + ' words')
for word, next_words_dict in bigram_words_counts.items():
    for nextword in matching_words_set:
        bigram_words_counts[word][nextword] = next_words_dict.get(nextword,0) * 1.4



# %% --- construct graph --------------------------------------------
# say we want a sentence of 7 words
# we have to get from start "" node to the 'endofline' in 7 moves
# so make a network with:
#  1 node layer ("") -> 2193 node word layer (repeat 7 layers for 7word) --> 1 node layer 'end of line'

# switched from networkx (implemented in python) to graph-tool as 12.33s-->0.2s for shortest path (on google)
# https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages

from tqdm import tqdm
from itertools import islice
from igraph import *
from math import log

n_words = 7
# edge_tuple_list used to create graph is 
# a array of tuples ie [('node1', 'node2', float_weight), ...]
edge_tuple_list = []

# add weights as weight = 1/freq 
# layersource to layer0
no_out_connections = {'', 'endofline', 'endofparagraph','dofline'}
orig_word_dict = bigram_words_counts.get('')
for word in all_words:
    # don't want our random to walk to end in a endofline early
    if(word in no_out_connections): continue

    node_key = "0_" + word
    # weight = log(1/orig_word_dict[word])+ 10
    weight = 1/orig_word_dict[word]
    edge_tuple = ('', node_key, weight)
    edge_tuple_list.append(edge_tuple)
    
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
            if(right_word in no_out_connections): continue
            right_node_key = str(i+1) + "_" + right_word
            # weight = log(1/orig_word_dict[right_word]) + 10
            weight = 1/orig_word_dict[right_word]
            edge_tuple = (left_node_key, right_node_key, weight)
            edge_tuple_list.append(edge_tuple)
        
# layer 6 is a closing layer
for left_word in all_words:
    if(left_word in no_out_connections): continue

    orig_word_dict = bigram_words_counts.get(left_word)
    left_node_key = str(n_words-1) + "_" + left_word
    # weight = log(1/orig_word_dict['endofline'])+ 10
    weight = 1/orig_word_dict['endofline']
    edge_tuple = (left_node_key, 'endofline', weight)
    edge_tuple_list.append(edge_tuple)

print('Graph conversion occuring')
g = Graph.TupleList(edge_tuple_list, weights=True, directed=True)
print('Graph made')

# %% --- shortest random walks --------------------------------------
from pqdict import pqdict

def get_weight(g, path):
    weight = 1
    for i in range(len(path)-1):
        # edge_weight = g[path[i], path[i+1]]
        edge_weight = g.es[g.get_eid(path[i], path[i+1], error=False)]['weight']
        weight = weight*edge_weight
    return weight

def get_string(g, path):
    string_arr = []
    for path_member in path:
        key = g.vs['name'][path_member]
        word = key[2:]
        word = re.sub(r"\u2005", " ", word) #forgot to take these out first
        word = re.sub(r"\u205f", " ", word) #forgot to take these out first
        if(key!=''): 
            string_arr =string_arr + [word]
    string_arr[0] = string_arr[0].capitalize()

    path_string = " ".join(string_arr)
    return path_string

print('Trying a bunch of random paths:')
minpq_dict = {}
all_paths= []
for i in tqdm(range(500000)):
    path = g.random_walk('',n_words+1, mode='out', stuck='return')
    weight = get_weight(g, path)
    all_paths.append(path)
    minpq_dict[i] = weight
minpq = pqdict(minpq_dict)
top_path_ids = list(minpq.popkeys())[:10]
top_strings = [get_string(g, all_paths[index]) for index in top_path_ids]
print('Best random walk paths:')
print(top_strings)

# %% --- copying is super fast so let's see how fast dropping nodes is ---------------------------------------------------------
import random

d = g.copy()
for i in range(n_words):
    begining = str(i)+'_'
    nodes_in_col = [v.index for v in d.vs if v['name'][:2]==begining]
    # https://www.geeksforgeeks.org/randomly-select-n-elements-from-list-in-python/
    count_dropping = int(0.01*len(nodes_in_col))
    nodes_to_drop = random.sample(nodes_in_col, count_dropping)
    d.delete_vertices(nodes_to_drop)
path = d.get_shortest_paths('', to='endofline', weights='weight', mode='out', output='vpath')[0]

path_string = get_string(g, path)
print(path_string)


