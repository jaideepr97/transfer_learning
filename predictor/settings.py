import os
import pandas as pd
import nltk

default_path = "/home/jaideeprao/Desktop/finalStretch"
# default_path = "/Users/adityadesai/Desktop/transfer_learning"

no_of_w2v_iterations = 3

# Main_Path = os.path.join(default_path, 'data')
# os.chdir(Main_Path)
#
# print("Loading data for w2v model")
# df1=pd.read_csv('movie_reviews_data/test-pos.csv')
# df2=pd.read_csv('movie_reviews_data/test-neg.csv')
# df3=pd.read_csv('movie_reviews_data/train-pos.csv')
# df4=pd.read_csv('movie_reviews_data/train-neg.csv')
#
# stop_words = set(nltk.corpus.stopwords.words('english'))
# stop_words.add("I")
# x1=df1['Reviews'].values.tolist()
# x2=df2['Review'].values.tolist()
# x3=df3['Review'].values.tolist()
# x4=df4['Review'].values.tolist()
# x = x1 + x2 + x3 + x4
# tok_corp = [nltk.word_tokenize(sent) for sent in x]
# print("Loaded")
