# %% --- Import libraries and json file -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from tqdm import tqdm
plt.style.use('ggplot')

processed_song_df = pd.read_csv('temp/phrase_fix_1.csv', index_col=0)
original_df = pd.read_csv('temp/processed_song_df.csv', index_col=0)
data = processed_song_df['0'].tolist()
data = [re.sub(r"endofparagraph", "", phrase) for phrase in data] # remove end of paragraph
data = [re.sub(r"endofline", "", phrase) for phrase in data] # remove end of phrase
orig_lyrics_list = data

# %% --- NMF --------------------------------------------------------
# NMF and LDA based on following:
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

categories = 5
no_top_words = 20

# Display function
def display_topics(model, feature_names, no_top_words):
    for i, topic in enumerate(model.components_):
        print("Topic %d:" % i)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# NMF
# tf-idf represenation
tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)[\w']*") # token pattern is because we include ' in our tokens
tfidf = tfidf_vectorizer.fit_transform(orig_lyrics_list)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


reconstruct_err = []
for alpha_index in tqdm(np.arange(0,1,0.2)):
    for l1_ratio in np.arange(0,1,0.2):
        for categs in range(2,15):
            nmf = NMF(n_components=categs, random_state=1, alpha=alpha_index, l1_ratio=l1_ratio, init='nndsvd')
            nmf.fit(tfidf)
            err = nmf.reconstruction_err_
            reconstruct_err.append({'err': err, 'alpha_index': alpha_index, 'l1_ratio': l1_ratio, 'categs': categs})
df = pd.DataFrame(reconstruct_err)

# %% --- Graph --------------------------------------------------------
errors = df['err']
plt.scatter(x=range(len(errors)), y=errors, c=df['categs'], cmap='Wistia')
cbar = plt.colorbar()
cbar.set_label('number of categories', rotation=270, labelpad=10)
plt.xlabel('Sklearn NMF run')
plt.ylabel('Error')
plt.savefig('../imgs/error_nmf.pdf')
plt.show()

# print the topics for n=5
nmf = NMF(n_components=categories, random_state=1, alpha=.1, l1_ratio=.4, init='nndsvd')
nmf.fit(tfidf)
display_topics(nmf, tfidf_feature_names, no_top_words)