# %% --- import lyrics ----------------------------------------------
import pandas as pd

new_lyrics = pd.read_csv('temp/phrase_fix_1.csv', index_col=0)['0'].to_list()

# %% --- calculate line lengths -------------------------------------
end = {'endofline', 'endofparagraph'}
past_lengths = []
current_length = 0
for song in new_lyrics:
    for word in song.split(' '):
        if word in end: 
            past_lengths.append(current_length)
            current_length = 0
        else: current_length+=1

# %% --- plot line lengths ------------------------------------------
import matplotlib.pyplot as plt
bins=18
plt.hist(past_lengths, bins=bins)
plt.show()

# %% --- fit normal curve to line lengths ---------------------------
# https://www.kite.com/python/answers/how-to-fit-a-distribution-to-a-histogram-in-python
from scipy import stats
import numpy as np

x_axis = np.arange(0, 20, 0.01)
mu, sigma = stats.norm.fit(past_lengths)
# stats.norm.pdf
best_fit_line = stats.norm.pdf(x_axis, mu, sigma)*len(past_lengths)
plt.hist(past_lengths, bins=bins)
plt.plot(x_axis, best_fit_line)
plt.show()
# %%
