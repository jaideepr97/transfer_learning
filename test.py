# import os
# import settings
# import pandas as pd
# import nltk
# import gensim
#
# Main_Path = os.path.join(settings.default_path, 'data')
# os.chdir(Main_Path)
#
# print("Loading data for w2v model in test")
# df1 = pd.read_csv('movie_reviews_data/test-pos.csv')
# df2 = pd.read_csv('movie_reviews_data/test-neg.csv')
# df3 = pd.read_csv('movie_reviews_data/train-pos.csv')
# df4 = pd.read_csv('movie_reviews_data/train-neg.csv')
#
# x1=df1['Reviews'].values.tolist()
# x2=df2['Review'].values.tolist()
# x3=df3['Review'].values.tolist()
# x4=df4['Review'].values.tolist()
# x = x1 + x2 + x3 + x4
# tok_corp = [nltk.word_tokenize(sent) for sent in x]
# print("Loaded")
#
# print("Creating w2v model in test")
# model = gensim.models.Word2Vec(tok_corp, min_count=1)
# print("Created")
#
# model.save("w2vmodel")

# loaded_model = gensim.models.Word2Vec.load("w2vmodel")
# print(loaded_model.predict_output_word(["great"]))

from predictor.training import classify

classify()
