# %% --- Import libraries and json file -----------------------------
import json
import matplotlib.pyplot as plt
import re
plt.style.use('ggplot')


json_file = open('../data_collection/data/song_lyrics.json')
song_lyrics_list = json.load(json_file)

# %% --- Create dataframe of song data ------------------------------
import pandas as pd

# construct dataframe with 1 row per song of:
# song  .... its name
# verse_lyrics .... self explanatory
# bridge_lyrics, chorus_lyrics, outro_lyrics
# song order .... Verse 1, Verse 2, Chorus, Verse 3 etc
processed_song_list = []
song_obj = song_lyrics_list[0]
for song_obj in song_lyrics_list:
    if song_obj['lyrics'] == '': continue
    obj = {}
    obj['song'] = song_obj['song']

    text = song_obj['lyrics']
    # remove initial \n
    while text[0] == '\n':
        text = text[1:]
    verse_markers = re.findall(r'\[(\w+[ \w]*)\]', text)
    obj['song_order'] = verse_markers
    # ', '.join(verse_markers) if you wanna make this a string

    # split lyrics by the markers we just found .. regex also includes []
    obj['verse_lyrics'] = ''
    obj['bridge_lyrics'] = ''
    obj['chorus_lyrics'] = ''
    obj['outro_lyrics'] = ''
    obj['all_lyrics'] = ''

    # https://stackoverflow.com/questions/10974932/split-string-based-on-a-regular-expression
    lyrics_arr = re.split(r'\[\w+[ \w]*\]', text)[1:] # start at 1 as index 0 is ''
    for verse_name, lyric_segment in zip(verse_markers, lyrics_arr):
        # clean up starts of lyrics
        while lyric_segment[0].isalpha() != True:
            lyric_segment = lyric_segment[1:]
        
        # treat verses
        if(verse_name[:5] ==  'Verse'):
            obj['verse_lyrics'] = obj['verse_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Bridge'):
            obj['bridge_lyrics'] = obj['bridge_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Chorus'):
            obj['chorus_lyrics'] = obj['chorus_lyrics'] + ' ' + lyric_segment
        elif(verse_name == 'Outro'):
            obj['outro_lyrics'] = obj['outro_lyrics'] + ' ' + lyric_segment
        
        # dump all lyrics with ' | ' between the verses
        obj['all_lyrics'] = obj['all_lyrics'] + ' | ' + lyric_segment

    # append to list
    processed_song_list.append(obj)


processed_song_df = pd.DataFrame(processed_song_list)
processed_song_df.to_csv('temp/processed_song_df.csv')

# %% --- Fix up the lyrics ------------------------------------------
from nltk.corpus import stopwords

orig_lyrics_list = processed_song_df['all_lyrics'].to_list()

# fix | and \n and ,  
orig_lyrics_list = [i.replace('|',' ') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace('\n',' ') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace(',',' ') for i in orig_lyrics_list]

# try to remove stopwords (clearer what is going on)
lyrics_list = [i.split(' ') for i in orig_lyrics_list]
english_stopwords = stopwords.words('english')
for c, lyrics in enumerate(lyrics_list):
    lyrics_list[c] = [i for i in lyrics if (i!='' and i not in english_stopwords)]
    orig_lyrics_list[c] = ' '.join(lyrics_list[c]).lower()

# fix - and '
orig_lyrics_list = [i.replace('-','') for i in orig_lyrics_list]
orig_lyrics_list = [i.replace('-','') for i in orig_lyrics_list]




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

nmf = NMF(n_components=categories, random_state=1, alpha=.1, l1_ratio=.4, init='nndsvd')
nmf.fit(tfidf)
# print the topics
display_topics(nmf, tfidf_feature_names, no_top_words)


# %% --- LDA --------------------------------------------------------
# LDA 
# bag of words
tf_vectorizer = CountVectorizer(token_pattern="(?u)[\w']*")
tf = tf_vectorizer.fit_transform(orig_lyrics_list)
tf_feature_names = tf_vectorizer.get_feature_names()
# create lda
# learning decay (0.5, 1.0] ... default .7
# learning_offset - greater than 1
lda = LatentDirichletAllocation(n_components=categories, learning_method='online', 
                                learning_offset=50 , learning_decay=.9, random_state=0, batch_size=10, max_iter=30)
lda.fit(tf)
print(lda.bound_)

# print the topics
print('------------------------------------------')
display_topics(lda, tf_feature_names, no_top_words)



# %% try gensim LDA model as has coherence
# based on following (large edits needed):
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# %% Clean data
# convert all to lower case and clean
processed_song_df['processed_lyrics'] = processed_song_df['all_lyrics'].map(lambda x: re.sub('[,\.!?]', '', x))
processed_song_df['processed_lyrics'] = processed_song_df['processed_lyrics'].map(lambda x: x.lower())
processed_song_df['processed_lyrics'].head()

# %% tokenise
import gensim
from gensim.utils import simple_preprocess
# note this doesn't keep '
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = processed_song_df.processed_lyrics.values.tolist()
data_words = list(sent_to_words(data)) # list of words, in list for each song

print(data_words[:1][0][:20])# %%

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])



# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# %%
import spacy

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1][0][:30])


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)



from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()



# %% Search around - taks 15mins
import numpy as np
import tqdm

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('./results/lda_tuning_results.csv', index=False)
    pbar.close()

# %% 

num_topics = 7
alpha = .01
beta = .91

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        alpha=0.01,
                                        eta=0.91)





# %%

import pyLDAvis.gensim
import pickle 
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('./results/ldavis_tuned_'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_tuned_'+ str(num_topics) +'.html')

LDAvis_prepared
# %%
