# %% --- Import libraries and json file -----------------------------
import json
from operator import index
import matplotlib.pyplot as plt
import re
import pandas as pd
plt.style.use('ggplot')

processed_song_df = pd.read_csv('temp/phrase_fix_1.csv', index_col=0)
original_df = pd.read_csv('temp/processed_song_df.csv', index_col=0)


# %% try gensim LDA model as has coherence
# based on following (large edits needed):
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

# %% tokenise
import gensim
from gensim.utils import simple_preprocess

# note this doesn't do anything as already processed lol
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = processed_song_df['0'].tolist()
data = [re.sub(r"endofparagraph", "", phrase) for phrase in data] # remove end of paragraph
data = [re.sub(r"endofline", "", phrase) for phrase in data] # remove end of phrase
data_words = list(sent_to_words(data)) # list of words, in list for each song


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

# %% --- Set up global variables ------------------------------------
import spacy

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# print(data_lemmatized[:1][0][:30])


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


#%% --- Basic LDA model ---------------------------------------------
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10, 
                                       random_state=100, chunksize=100, passes=10,
                                       per_word_topics=True)



from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score for basic LDA model: ', coherence_lda)


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k, 
                                           random_state=100, chunksize=100, passes=10,
                                           alpha=a, eta=b)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()



#%% --- Search around - takes 15mins --------------------------------
import numpy as np
import tqdm

lda_tuning_results = pd.read_csv('./results/lda_tuning_results.csv')

# grid = {}
# grid['Validation_Set'] = {}

# # Topics range
# min_topics = 2
# max_topics = 11
# step_size = 1
# topics_range = range(min_topics, max_topics, step_size)

# # Alpha parameter
# alpha = list(np.arange(0.01, 1, 0.3))
# alpha.append('symmetric')
# alpha.append('asymmetric')

# # Beta parameter
# beta = list(np.arange(0.01, 1, 0.3))
# beta.append('symmetric')

# # Validation sets
# num_of_docs = len(corpus)
# corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
#                corpus]

# corpus_title = ['75% Corpus', '100% Corpus']

# model_results = {'Validation_Set': [],
#                  'Topics': [],
#                  'Alpha': [],
#                  'Beta': [],
#                  'Coherence': []
#                 }



# # Can take a long time to run
# if 1 == 1:
#     pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
#     # iterate through validation corpuses
#     for i in range(len(corpus_sets)):
#         # iterate through number of topics
#         for k in topics_range:
#             # iterate through alpha values
#             for a in alpha:
#                 # iterare through beta values
#                 for b in beta:
#                     # get the coherence score for the given parameters
#                     cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
#                                                   k=k, a=a, b=b)
#                     # Save the model results
#                     model_results['Validation_Set'].append(corpus_title[i])
#                     model_results['Topics'].append(k)
#                     model_results['Alpha'].append(a)
#                     model_results['Beta'].append(b)
#                     model_results['Coherence'].append(cv)
                    
#                     pbar.update(1)
#     lda_tuning_results = pd.DataFrame(model_results)
#     lda_tuning_results.to_csv('./results/lda_tuning_results.csv', index=False)
#     pbar.close()

plt.scatter(lda_tuning_results['Topics'], lda_tuning_results['Coherence'])
plt.xlabel('Topics')
plt.ylabel('Coherence')
plt.title('Topics vs Coherence for different LDA models')
plt.show()



# %% --- Run champion model -----------------------------------------
num_topics = 7
alpha = 0
beta = .91

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        alpha=0.01,
                                        eta=0.91)

# %% --- Save the champ data ---------------------------------------
import json

rows = []
for i in range(len(corpus)):
    arr = lda_model[corpus[i]]

    #calculate champ
    tuple_champ = (0,0)
    for categ_tuple in arr:
        categ, prob = categ_tuple
        if(prob>tuple_champ[1]): tuple_champ = categ_tuple
    
    rows.append({'categ': tuple_champ[0], 'p': tuple_champ[1]})

df = pd.DataFrame(rows)
df['song'] = original_df['song']
df.to_csv('temp/lda_categ_song.csv')


# winning_id was 1 (if start numbering at 1) ... for romantic like songs
winning_id = 1
json_ouptut = lda_model.show_topic(topicid=winning_id, topn=200)
# stupdi conversion of datatypes
json_ouptut = [(str(word), float(p)) for word, p in json_ouptut] 
with open('./temp/lda_words_categ1.json', 'w') as outfile:
    json.dump(json_ouptut, outfile)


# %% --- Visualize the topics ---------------------------------------
import os
import pyLDAvis.gensim
import pickle 
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_tuned_'+ str(num_topics) +'.html')

LDAvis_prepared
# %%
