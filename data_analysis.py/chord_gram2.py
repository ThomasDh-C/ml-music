# %% Import json file
import json
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np

json_file = open('../data_collection/data/chord_progs.json')
pages_to_search = json.load(json_file)

key_count = dict()
intro_chords = []
verse_chords = []
prechorus_chords = []
chorus_chords = []
bridge_chords = []
for page in pages_to_search:
    key_obj=page['key']
    # print(key_obj)
    if (key_count.get(key_obj) != None):
        count = key_count.get(key_obj)
        key_count[key_obj] = count + 1
    else:
        key_count[key_obj] = 1
    for ichord_obj in page['intro prog']:
        chord_num = ichord_obj['num']
        intro_chords.append(chord_num)
    for vchord_obj in page['verse prog']:
        chord_num = vchord_obj['num']
        verse_chords.append(chord_num)
    for pchord_obj in page['pre-chorus prog']:
        chord_num = pchord_obj['num']
        prechorus_chords.append(chord_num)
    for cchord_obj in page['chorus prog']:
        chord_num = cchord_obj['num']
        chorus_chords.append(chord_num)
    for bchord_obj in page['bridge prog']:
        chord_num = bchord_obj['num']
        bridge_chords.append(chord_num)

champ = 0
for k in key_count:
    if (key_count.get(k) > champ):
        champ = key_count.get(k)
        champ_key = k

print("Key: " + champ_key)

# Thomas's trigram implementation - INTRO
# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
iall_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
ingram_words_counts = {}
for song in intro_chords:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in ingram_words_counts: ingram_words_counts[prev_word] = {}
        count = ingram_words_counts[prev_word].get(word,0)
        ingram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        iall_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    
# laplace smooth
for word, next_words_dict in ingram_words_counts.items():
    for nextword in iall_words:
        count = next_words_dict.get(nextword,0)
        ingram_words_counts[word][nextword] = count + .3
iunknown_dict = {word: .3 for word in iall_words}

# Thomas's trigram implementation - VERSE
# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
vall_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
vngram_words_counts = {}
for song in verse_chords:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in vngram_words_counts: vngram_words_counts[prev_word] = {}
        count = vngram_words_counts[prev_word].get(word,0)
        vngram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        vall_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    
# laplace smooth
for word, next_words_dict in vngram_words_counts.items():
    for nextword in vall_words:
        count = next_words_dict.get(nextword,0)
        vngram_words_counts[word][nextword] = count + .3
vunknown_dict = {word: .3 for word in vall_words}

# Thomas's trigram implementation - PRE-CHORUS
# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
pall_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
pngram_words_counts = {}
for song in prechorus_chords:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in pngram_words_counts: pngram_words_counts[prev_word] = {}
        count = pngram_words_counts[prev_word].get(word,0)
        pngram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        pall_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    
# laplace smooth
for word, next_words_dict in pngram_words_counts.items():
    for nextword in pall_words:
        count = next_words_dict.get(nextword,0)
        pngram_words_counts[word][nextword] = count + .3
punknown_dict = {word: .3 for word in iall_words}

# Thomas's trigram implementation - CHORUS
# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
call_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
cngram_words_counts = {}
for song in chorus_chords:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in cngram_words_counts: cngram_words_counts[prev_word] = {}
        count = cngram_words_counts[prev_word].get(word,0)
        cngram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        call_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    
# laplace smooth
for word, next_words_dict in cngram_words_counts.items():
    for nextword in call_words:
        count = next_words_dict.get(nextword,0)
        cngram_words_counts[word][nextword] = count + .3
cunknown_dict = {word: .3 for word in iall_words}

# Thomas's trigram implementation - BRIDGE
# %% --- I guess that kinda works ... let's see if better luck with trigrams
n=2 #length of previous words in memory
ball_words=set()
# word: {'next1a next1b': count, 'next2a next2b': count}
bngram_words_counts = {}
for song in bridge_chords:
    song_arr = [word for word in song.split(' ') if word!='']

    prev_word_arr = [''] * n
    prev_word = " ".join(prev_word_arr)
    # iterate through all song words
    for word in song_arr:
        if prev_word not in bngram_words_counts: bngram_words_counts[prev_word] = {}
        count = bngram_words_counts[prev_word].get(word,0)
        bngram_words_counts[prev_word][word] = count+1

        prev_word_arr = prev_word_arr[1:] + [word]
        prev_word = " ".join(prev_word_arr)
        ball_words.add(word)
    # song should end on prev_word_arr = [random, endofparagraph]
    
# laplace smooth
for word, next_words_dict in bngram_words_counts.items():
    for nextword in ball_words:
        count = next_words_dict.get(nextword,0)
        bngram_words_counts[word][nextword] = count + .3
bunknown_dict = {word: .3 for word in iall_words}

# %% --- run ngram model - INTRO
from copy import deepcopy
import re

prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(4):
    # add in some randomness
    p_dist = {}
    if prev_word in ingram_words_counts.keys():
        p_dist = ingram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(iunknown_dict) # if unseen use unknown dict
    
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

# %% --- run ngram model - VERSE
from copy import deepcopy
import re

prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(4):
    # add in some randomness
    p_dist = {}
    if prev_word in vngram_words_counts.keys():
        p_dist = vngram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(vunknown_dict) # if unseen use unknown dict
    
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

# %% --- run ngram model - PRECHORUS
from copy import deepcopy
import re

prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(4):
    # add in some randomness
    p_dist = {}
    if prev_word in pngram_words_counts.keys():
        p_dist = pngram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(punknown_dict) # if unseen use unknown dict
    
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

# %% --- run ngram model - CHORUS
from copy import deepcopy
import re

prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(4):
    # add in some randomness
    p_dist = {}
    if prev_word in cngram_words_counts.keys():
        p_dist = cngram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(cunknown_dict) # if unseen use unknown dict
    
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

# %% --- run ngram model - BRIDGE
from copy import deepcopy
import re

prev_word_arr = [''] * n
prev_word = " ".join(prev_word_arr)
words = []
for i in range(4):
    # add in some randomness
    p_dist = {}
    if prev_word in bngram_words_counts.keys():
        p_dist = bngram_words_counts.get(prev_word)
    else:
        p_dist = deepcopy(bunknown_dict) # if unseen use unknown dict
    
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