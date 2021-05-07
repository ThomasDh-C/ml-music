import pandas as pd
import re
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# %% --- import stats, string views --> int view count --------------
views_df = pd.read_csv('../data_collection/data/song_stats.csv', index_col=0)
viewcount = views_df['viewcount']
views = []
for row in viewcount:
    rowstr = row[:-6]
    rowstr = re.sub(r",", "", rowstr)
    views.append(int(rowstr))
views_df['int_viewcount'] = views

# get rid of rows that are commentary
# https://stackoverflow.com/questions/28679930/how-to-drop-rows-from-pandas-data-frame-that-contains-a-particular-string-in-a-p
views_df = views_df[~views_df['songname'].str.contains("Commentary")]

# clean up song titles
songname = views_df['songname']
names = []
for row in songname:
    # remove Taylor Swift - at the start
    if row[:15]=='Taylor Swift - ': row = row[15:]
    # remove brackets:
    brackets = re.search(r' \(', row)
    if brackets:
        first = brackets.span()[0]
        row = row[:first]
    # convert to lower case
    row = row.lower()
    names.append(row)
views_df['fixed_songname'] = names

# now a lot of songs will have the same title, so create a new df of the maxes
max_views_per_song = []
unique_song_names = views_df['fixed_songname'].unique()
for song_name in unique_song_names:
    small_df = views_df[views_df['fixed_songname'] == song_name]
    max_views = max(small_df['int_viewcount'])

    # print(song_name)
    row = {'fixed_songname': song_name, 'views': max_views}
    max_views_per_song.append(row)
reduced_views_df = pd.DataFrame(max_views_per_song)



# %% --- import lyrics ---------------------------------------------
lyrics_df = pd.read_csv('temp/lda_categ_song.csv', index_col=0)
# clean up song titles
songname = lyrics_df['song']
names = []
for row in songname:
    # remove Taylor Swift - at the start
    if row[:15]=='Taylor Swift - ': row = row[15:]
    # remove brackets:
    brackets = re.search(r' \(', row)
    if brackets:
        first = brackets.span()[0]
        row = row[:first]
    # convert to lower case
    row = row.lower()
    names.append(row)
lyrics_df['fixed_songname'] = names



# %% --- merge the two
merged_dataset = lyrics_df.merge(reduced_views_df, on='fixed_songname', how='inner')
merged_dataset.loc[:,['song', 'categ', 'views']].to_csv('results/merged_song_views.csv')


# %% --- also import lda_tuning_results
lda_tuning_results = pd.read_csv('results/lda_tuning_results.csv')

# %% --- make some graphs ---
coherence = lda_tuning_results['Coherence']
labels = lda_tuning_results['Topics']
plt.scatter(x=range(len(coherence)),y=coherence, c=labels, cmap='Dark2')
cbar = plt.colorbar()
cbar.set_label('number of categories', rotation=270, labelpad=10)
plt.xlabel('Gensim LDA run')
plt.ylabel('Coherence')
plt.savefig('../imgs/coherence_lda.pdf')
plt.show()

# all songs success
plt.hist(merged_dataset['views'], bins=100)
plt.xlabel('Views on YouTube')
plt.ylabel('Song count')
plt.savefig('../imgs/views_by_song.pdf')
plt.show()

# topics vs success
plt.scatter(merged_dataset['categ'], merged_dataset['views'])
plt.yscale('log') #https://stackoverflow.com/questions/18773662/python-scatter-plot-logarithmic-scale
plt.xlabel('Topic')
plt.ylabel('Views on YouTube')
plt.savefig('../imgs/lda_views_by_topic.pdf')
plt.show()
# %%
