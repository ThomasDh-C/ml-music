# %% --- import lyrics ----------------------------------------------
import pandas as pd

new_lyrics = pd.read_csv('temp/phrase_fix_1.csv', index_col=0)['0'].to_list()

# %% --- calculate line lengths -------------------------------------
end = {'endofline', 'endofparagraph'}
past_lengths = [] # number of tokens per line
past_song_lengths = [] # number of lines needed
current_length = 0
song_length = 0
for song in new_lyrics:
    for word in song.split(' '):
        if word in end: 
            past_lengths.append(current_length)
            current_length = 0
            song_length += 1
        else: current_length+=1
    past_song_lengths.append(song_length)
    song_length = 0
# %% --- plot line lengths ------------------------------------------
import matplotlib.pyplot as plt
print('Number of tokens per line')
bins=18
plt.hist(past_lengths, bins=bins)
plt.show()

print('Number of lines per song')
bins=10
plt.hist(past_song_lengths, bins=bins)
plt.show()
print('------------------------------------------')
# %% --- fit normal curve to line lengths ---------------------------
# https://www.kite.com/python/answers/how-to-fit-a-distribution-to-a-histogram-in-python
from scipy import stats
import numpy as np

print('Fit normal to number of tokens per line')
bins = 18
x_axis = np.arange(0, 20, 0.01)

mu, sigma = stats.norm.fit(past_lengths)
best_fit_line = stats.norm.pdf(x_axis, mu, sigma)*len(past_lengths)

plt.hist(past_lengths, bins=bins)
plt.plot(x_axis, best_fit_line)
plt.show()
print('Mu: ' + str(mu))
print('Sigma: ' + str(sigma))


print('Fit normal to number of lines per song')
bins=10
x_axis = np.arange(10, 65, 0.01)

mu, sigma = stats.norm.fit(past_song_lengths)
# add in a fudge factor of 5 for graphing to make it pretty
best_fit_line = stats.norm.pdf(x_axis, mu, sigma)*len(past_song_lengths)*5 

plt.hist(past_song_lengths, bins=bins)
plt.plot(x_axis, best_fit_line)
plt.show()
print('Mu: ' + str(mu))
print('Sigma: ' + str(sigma))



# %%
