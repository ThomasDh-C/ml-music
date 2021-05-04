# for use for singular datum (ie, tempo, duration, 
# time signature) not influenced by other factors 
# besides viewcount

import os
import math

f = open("song_stats.csv", "r")

champ_stream = -math.inf

all_data = f.readlines()
for line in all_data:
    split_up = line.split(",")
    if (split_up[4] > champ_stream):
        champ_stream = split_up[4]
        champ_song = split_up[1]

# use spotify api to find:
# tempo
# duration
# time signature
# for champ_song