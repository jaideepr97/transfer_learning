import os
import pandas as pd
import nltk
import settings
import gensim

no_of_w2v_iterations = 3

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)

print("Loading data for w2v model")
df1=pd.read_csv('movie_reviews_data/test-pos.csv')
df2=pd.read_csv('movie_reviews_data/test-neg.csv')
df3=pd.read_csv('movie_reviews_data/train-pos.csv')
df4=pd.read_csv('movie_reviews_data/train-neg.csv')

stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add("I")
x1=df1['Reviews'].values.tolist()
x2=df2['Review'].values.tolist()
x3=df3['Review'].values.tolist()
x4=df4['Review'].values.tolist()
x = x1 + x2 + x3 + x4
tok_corp = [nltk.word_tokenize(sent) for sent in x]
print("Loaded")

print("Creating w2v model 0")
model = gensim.models.Word2Vec(tok_corp, min_count=1)
model.save("w2vmodel0")
print("Created")

j = 1
while j <= settings.no_of_w2v_iterations:
    print("Creating w2v model {}".format(j))
    model = gensim.models.Word2Vec(tok_corp, min_count=1)
    model.save("w2vmodel{}".format(j))
    print("Created")
    j = j + 1
