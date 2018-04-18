import gensim
import pandas as pd
import nltk
import re
import os
import settings
import pickle
from nltk.corpus import stopwords
returned_words = []

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)

# print("Loading data for w2v model in w2vGen")
# df1=pd.read_csv('movie_reviews_data/test-pos.csv')
# df2=pd.read_csv('movie_reviews_data/test-neg.csv')
# df3=pd.read_csv('movie_reviews_data/train-pos.csv')
# df4=pd.read_csv('movie_reviews_data/train-neg.csv')
#
# stop_words = set(stopwords.words('english'))
# stop_words.add("I")
# x1=df1['Reviews'].values.tolist()
# x2=df2['Review'].values.tolist()
# x3=df3['Review'].values.tolist()
# x4=df4['Review'].values.tolist()
# x = x1 + x2 + x3 + x4
# tok_corp = [nltk.word_tokenize(sent) for sent in x]
# print("Loaded")

def word2vec_words(sent, i):
    # print("Loading model number {} for w2v in w2vGen".format(i))
    # model = gensim.models.Word2Vec(settings.tok_corp, min_count=1)
    model = gensim.models.Word2Vec.load("w2vmodel{}".format(i))
    # print("Loaded")
    x = sent
    filtered_sentence= re.sub("[^\w]", " ",  x).split()
    return model.predict_output_word(filtered_sentence)
