'''
Calculate similarity using BOW and pre-trained embedding
'''
from gensim.test.utils import datapath                                  
from gensim.models import KeyedVectors 
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer

wv = KeyedVectors.load_word2vec_format(datapath("/users/lgaray/resources/PubMed-w2v.bin"), binary=True)

# using embeddings
def emb(s1):
    summ = np.zeros(200)
    for tok in s1:
        try:
            summ += wv.get_vector(tok)
        except: 
            pass
    summ = summ.reshape(1, summ.shape[0])
    return summ

# using bow
vect = CountVectorizer()
def bow(s1, s2):
    x = ' '.join(s1)
    y = ' '.join(s2)
    try:
        x = vect.fit_transform([x])
        y = vect.fit_transform([y])
    except:
        x = np.zeros(0)
        y = np.zeros(0)
    if x.shape > y.shape:
        y.resize(x.shape)
    else:
        x.resize(y.shape)
    return (x,y)

def sim(s1, s2):
    # embedding
    v1 = emb(s1)
    v2 = emb(s2)
    res = cosine_similarity(v1, v2)
    # bow
    x1, x2 = bow(s1, s2)
    if x1.shape[0] == 0:
        resbowCV = [[0]]
    else:
        resbowCV = cosine_similarity(x1, x2)
    return (res[0][0], resbowCV[0][0])

