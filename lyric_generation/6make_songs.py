# %% --- get json file --------------------------------------
import json

json_file = open('./temp/phrases_graph_model_ouput.json')
phrase_model = json.load(json_file)

# %% --- generate song --------------------------------------
import numpy as np
import re

# lines_per_song~N(44.66, (11.40)^2)
mu, sigma = 44.66, 11.40 
song_lines = int(np.random.normal(mu, sigma))

# words_per_line~N( 7.50, (2.88)^2)
mu, sigma = 7.50, 2.88 

line_lengths = []
while len(line_lengths)<song_lines:
    line_length = int(np.random.normal(mu, sigma))
    # in the set of song lyrics we have
    if(line_length>2 and line_length<18 and line_length%2==0):
        line_lengths.append(line_length)

random_walk_song = []
shortest_path_song = []
for line_length in line_lengths:
    line_length = str(line_length)
    random_phrase_list = phrase_model[line_length]['random_phrases']
    shortest_phrase_list = phrase_model[line_length]['shortest_phrases']

    length_random_phrase = len(random_phrase_list)
    random_phrase_index = np.random.randint(0,length_random_phrase)
    random_walk_song.append(random_phrase_list[random_phrase_index])

    length_shortest_phrase = len(shortest_phrase_list)
    shortest_phrase_index = np.random.randint(0,length_shortest_phrase)
    shortest_path_song.append(random_phrase_list[shortest_phrase_index])

# some more cleanup left
random_walk_song = [re.sub(r"_", "", phrase) for phrase in random_walk_song]
shortest_path_song = [re.sub(r"_", "", phrase) for phrase in shortest_path_song]

random_walk_song = "\n".join(random_walk_song)
shortest_path_song = "\n".join(shortest_path_song)

# %% --- print song --------------------------------------

print('Random walk song:')
print(random_walk_song)
print('-------------------')
print('Shortest path song:')
print(shortest_path_song)


# An example of our quality output:

# Random walk song:
# Med drive rebekah longest time were leaving shop sadness catch
# Twice sweet tea avenue six happy know i still
# Those bless my town an lasts
# Pillow to the christmas
# Suit through real marching india art hand then i talk
# There will shine phones count waitin'
# I around the ghostly drag
# I send fresh holds crying bury band cynical
# As loudest you will home waitress
# What could little money mythical bets wherever summers suspicious
# You leave play side bones meet
# Never would toast disbelief spring wondrous brighter guitar
# Keep cursing fairytale endofparagraph[prechorus] face photographs
# I grab facts low tall even know twenties
# Gym champions folk song like thеn to win
# Pools changin' home on some neither
# On the step farm make white's waking hollow untouchable jeans
# It all heroes you just skeletons tattoo counted ships passenger's sensible
# I see ship sequin throwing gone
# I serve slow stronger drawing lifetimes
# It rains crescent older flushed heels saw above angry haunt _the bus _gone
# Waters a boy dancing screaming spots gets but i younger
# If it horse swear
# Ships perfеct contrarian patch lakes words when you
# Say the little enemies yes packed will forgot
# My except word need is sweet
# Like flying make the words wishes frozen my louis
# I spend battle outnumbered forever traveled midnight telling day bleed
# You still broken everybody sink places
# I search clandestine picked am hear realized chill gold windermere
# I played dolls leaves scrap hollow
# I know coast took her space uppеr coast well shit passed away _read _clover
# The night makeup invisible tuesday forth new pocket twenties queens
# Mom's ring i know dorothea
# It expired near care talks wives forgotten story see what
# Salt settle head that to leave shade on ground james
# Small to word she went somehow
# Vintage kiss legs religion deep to
# Anyone heads very our noose i want whisper staring beyond
# I acted hanging 2 hell adore
# Hope of pull lookout
# January outdoor crestfallen wool moments get into a
# My side sleep will keep younger wakes shade such
# It even eye accidents scorpion secret fifty haunt
# And they bags teaches gleaming she he asks
# My heartbeat line ran
# Bone pages are bill to bright wool i would crossing _crescent _meets
# What they easy wandered
# I saw picking fun
# Sunday flannel in my middleclass falls
# -------------------
# Shortest path song:
# First few persist the and that changed madhouse started fault
# I stare pull screen's sixteen my dying doing all
# There will shine phones count waitin'
# My was just feelings
# Signal sixteen with the a bad terrified marching same glance
# Cynical window woah return in the
# Mom's ring i know dorothea
# You all eclipsed note care folk mother myself
# I i middleclass dreamer sharе tasteful
# I knew killing blur fade write airport assumptions
# I still grace field desk radio
# I will dear tshirt money thorns weeks untouchable
# I serve slow stronger drawing lifetimes
# I think dappled magnetic sleepless tricks clues existed
# I break thing rolling senior devils tupelo pérignon
# I still grace field desk radio
# Guy heels secret holds yours want to buy full football
# Big ours backlogged stalk accidents our love you wishes park
# I just went hell color of feels
# You tell perfеct ohohohoh sick fake
# I used leader choices the summer than knelt near anymore nice _uhuh _get you out
# Back will be daughter chances dalí answer used to think angel heartbreak
# Each i know throwing
# Think taylor never mine spin 'round insincere wonderful
# I eyes hustled cornelia spinning truth climb watching
# I caught was charming page gardens feet
# I stare pull screen's sixteen my dying doing all
# Suit through real marching india art hand then i talk
# I pulled seem remember shiniest pumpkin
# First few persist the and that changed madhouse started fault
# You put drop small about everything
# I dreamed were throwing parties wonder stranger lalalala standard rules lie _skeletons _heart
# You would juliet very west smart nose unnecessary sidewalk white's
# Sadness in breakable hear
# In spots january the floor meetings and longing models always sense
# You think sense you'rе else usually naïve sparks cascade outskirts
# Cynical window woah return in the
# I i was hang barren look streets
# I there was bleedin' asked fields sweater dagger scorpion because
# I told smell likes been rooms
# Noose with me well
# Gray drivin' trace behind today try try stephen
# I break thing rolling senior devils tupelo pérignon
# I knew killing blur fade write airport assumptions
# Avenue summer truck still take the think it all
# I do ahahah homeroom
# Wreck superstar used blue maddest king's brought it is sacred _ghostly _temple
# You bless listening kid
# I saw picking fun
# Make weapons teal enough i feel