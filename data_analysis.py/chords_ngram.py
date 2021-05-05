# %% Import json file
import json
import nltk
# nltk.download()
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np

note_arr = ["A", "B", "C", "D", "E", "F", "G", "A", "B", "C", "D", "E", "F", "G"]

json_file = open('../data_collection/data/chords.json')
pages_to_search = json.load(json_file)

new_lyrics = []
all_chords = {}
for page in pages_to_search:
    
    for chord_obj in page['chords']:
        stripped = chord_obj['chord'].split('-')

        new_lyrics.append(stripped[1])

# Thomas's trigram implementation
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
import re

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

json_file.close()



































# # find all chords we have to find the sounds of

# chord_str = ""
# all_chords = {}
# for page in pages_to_search:
#     for chord_obj in page['chords']:
#         chord = chord_obj['chord']
#         new_count = all_chords.get(chord, 0) + 1
#         all_chords[chord] = new_count
#         stripped = chord_obj['chord'].split('-')
#         chord_str += stripped[1] + " "
# #print(chord_str)

# #token = nltk.word_tokenize(chord_str)
# token = chord_str.split()
# trigrams = ngrams(token,3)

# for t in trigrams:
#     print(t)

# #print(all_chords)
# # printed:
# # {u'label-E_min7': 51, u'label-G_sus4': 7, u'label-Ds_min7': 1, u'label-Eb_maj': 58, 
# # u'label-C_maj7': 7, u'label-E_maj': 93, u'label-G_min7': 12, u'label-G_5': 3, 
# # u'label-D_min7': 5, u'label-C_sus4': 3, u'label-B_maj': 162, u'label-As_maj': 1, 
# # u'label-F_maj': 405, u'label-Ab_maj': 50, u'label-G_min': 36, u'label-C_min': 11, 
# # u'label-Gb_maj': 32, u'label-C_maj': 711, u'label-Ds_maj': 1, u'label-Db_maj': 59, 
# # u'label-F_min': 12, u'label-Cs_min': 62, u'label-Fs_maj': 77, u'label-B_5': 1, 
# # u'label-A_maj': 231, u'label-E_min9': 19, u'label-E_7': 2, u'label-D_sus4': 6, 
# # u'label-Bb_maj': 63, u'label-G_maj': 849, u'label-B_min7': 7, u'label-A_5': 4, 
# # u'label-Cs_maj': 95, u'label-Ds_min': 46, u'label-Fs_min': 49, u'label-E_min': 233, 
# # u'label-G_maj7': 24, u'label-D_min': 104, u'label-C_5': 1, u'label-B_min': 107, 
# # u'label-A_min': 370, u'label-A_sus4': 6, u'label-D_maj': 602, u'label-Gs_min': 39, 
# # u'label-E_5': 3, u'label-Bb_min': 45, u'icon-rest': 115}

# # champ_chord = ""
# # champ_count = 0
# # for c in all_chords:
# #     if all_chords.get(c) > champ_count:
# #         champ_count = all_chords.get(c)
# #         champ_chord = c

# # print("start at chord " + str(champ_chord) + " with count " + str(champ_count))

# json_file.close()