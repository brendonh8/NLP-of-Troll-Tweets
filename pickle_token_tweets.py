#!/usr/bin/env python
# coding: utf-8

# ### Project 4
# ### Brendon Happ
# ### NLP

# In[202]:


import pandas as pd
import numpy as np
import os
import pickle
import re
from smart_open import smart_open

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities, matutils
from gensim.models.phrases import Phraser, Phrases
from gensim.models.ldamulticore import LdaMulticore


# In[2]:


full_df_list = []

for (dirname, dirs, files) in os.walk('clean_pickles'):
    for filename in files:
        with open(os.path.join('clean_pickles', filename), 'rb') as f:
            full_df_list.append(pd.read_pickle(f))


# In[3]:


full_df = pd.concat(full_df_list, axis=0, ignore_index=True)


# In[4]:


full_df.columns


# In[5]:


#words = set(stopwords.words("english"))


# In[6]:


#def remove_stopwords(df):
#    df = ' '.join(word for word in df.split() if word not in words)
#    return df


# In[7]:


#clean_col = full_df['content'].apply(remove_stopwords)
#full_df.loc[:, 'content'] = clean_col.values


# In[8]:


tweets = full_df.content


# ### Tokenization

# First, tokenize the documents, remove common words 

# In[43]:


tweets.shape


# In[10]:


nlp = spacy.load('en', disable=['parser', 'ner'])


# In[11]:


stemmer = SnowballStemmer("english")


# In[57]:


def preprocess_text(text):
    '''
    Tokenises, and lemmatize's using spacy. Returns a string of space seperated tokens.
    '''
    #words = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = nlp(text)
    stops = set(stopwords.words("english"))

    result = []
    global cache
    for word in words:
        # Memoization 
        if word in stops:
            continue
        elif len(word) == 1:
            continue
        elif len(word) == 2:
            continue
        elif word not in cache:
            lemma = str(word.lemma_) if word.lemma_ != "-PRON-" else str(word)
            cache[word] = lemma
        else:
            lemma = cache[word]
        result.append(lemma)
    return " ".join(result)


# In[130]:


tokenized_tweets = []
cache = {}
for tweet in tweets:
    tokenized_tweets.append(preprocess_text(tweet))

with open('token_tweets.pickle', 'wb') as f:
    pickle.dump(tokenized_tweets, f, protocol=pickle.HIGHEST_PROTOCOL)
# In[131]:


#def ngrams_split(lst, n):
#    return [' '.join(lst[i:i+n]) for i in range(len(lst)-n)]


# In[132]:


#bigram_tweets = []
#for tweet in tokenized_tweets:
#    bigram_tweets.append(ngrams_split(tweet.split(), 2))


# ### Topic/Concept Modeling

# To convert documents to vectors, use a **bag of words** representation. In this representation, each document is represented by one vector where a vector element i represents the number of times the ith word appears in the document.
# 
# It is advantageous to represent the questions only by their (integer) ids. The mapping between the questions and ids is called a dictionary:

# In[133]:


gensim_tweets = []
for tweet in tokenized_tweets:
    tweet_list = tweet.split()
    gensim_tweets.append(tweet_list)


# **Add bigrams to gensim formatted tweets**

# In[194]:


phrases = Phrases(gensim_tweets, min_count=3, threshold=100)
bigram = Phraser(phrases)


# In[195]:


bi_tweets = []
for sent in bigram[gensim_tweets]:
    bi_tweets.append(sent)

with open('bi_tweets.pickle', 'wb') as file:
    pickle.dump(bi_tweets, file, protocol=pickle.HIGHEST_PROTOCOL)
