!pip install gensim pymorphy2 -q

!pip install -U sentence-transformers

"""#Import necessary libraries"""

# Commented out IPython magic to ensure Python compatibility.
import re
import nltk
import pickle
import random


import gensim
import pymorphy2

import collections

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

from functools import lru_cache

from nltk.corpus import stopwords

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from sentence_transformers import SentenceTransformer

nltk.download('stopwords')

sw = stopwords.words('english')

morph = pymorphy2.MorphAnalyzer()

"""#Functions"""

def read_json(path):

    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)


def remove_stopwords(data):

    for i in range(len(data)):
        tmp = []

        for j in data[i]:
            if j not in sw:
                tmp.append(j)

        data[i] = tmp

        return data


def tokenize_text_simple_regex(txt, min_token_size=4, reg_exp=r'[\w\d]+'):
    txt = txt.lower()
    token_re = re.compile(reg_exp)
    all_tokens = token_re.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


def plot_vectors(vectors, labels, how='tsne', ax=None):
    if how == 'tsne':
        projections = TSNE().fit_transform(vectors)
    elif how == 'svd':
        projections = TruncatedSVD().fit_transform(vectors)

    x = projections[:, 0]
    y = projections[:, 1]

    ax.scatter(x, y)
    for cur_x, cur_y, cur_label in zip(x, y, labels):
        ax.annotate(cur_label, (cur_x, cur_y))


@lru_cache(100000)
def lemmatize(s):
    s = str(s).lower()
    return morph.parse(s)[0].normal_form

init_random_seed()

"""#Processing of train texts"""

train_texts = tokenize_corpus(train_texts)
train_texts = remove_stopwords(train_texts)
train_texts = [[lemmatize(word) for word in sent] for sent in train_texts]

"""#Processing of test texts"""

train_texts = tokenize_corpus(train_texts)
train_texts = remove_stopwords(train_texts)
train_texts = [[lemmatize(word) for word in sent] for sent in train_texts]

train_texts = [i for i in train_texts if len(i) > 15] # remove abstracts, which lenght < 15 tokens

# processing of train and test texts for transformer

train_texts_trans = [' '.join(i for i in abst) for abst in train_texts]
test_texts_trans = [' '.join(i for i in abst) for abst in test_texts]

"""#Model"""

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

embeddings_train = model.encode(train) # getting embeddings of train abstracts

embeddings_test = model.encode(test) # getting embeddings of test abstracts

train_matrix = embeddings_train

res = [] # indexes of articles that fit the customer
thr = 0.75 # suitability threshold

for i in range(len(embeddings_test)):

    relevance = np.matmul(train_matrix, embeddings_test[i])
    top_ind = (-relevance.flatten()).argsort()[:5]

    if relevance[top_ind].mean() > 0.75:
        res.append(i)

"""#Save model"""

with open('rec_sys_trans.pkl', 'wb') as f:
    pickle.dump(model, f)
