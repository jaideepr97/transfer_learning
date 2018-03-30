import gensim
import pandas as pd
import nltk
import re
import pickle
from nltk.corpus import stopwords
returned_words = []

#df1=pd.read_csv('movie_reviews_data/test-pos.csv'); 
#df2=pd.read_csv('movie_reviews_data/test-neg.csv');
#df3=pd.read_csv('movie_reviews_data/train-pos.csv');
#df4=pd.read_csv('movie_reviews_data/train-neg.csv');
#
#stop_words = set(stopwords.words('english'))
#stop_words.add("I")
#x1=df1['Reviews'].values.tolist()
#x2=df2['Review'].values.tolist()
#x3=df3['Review'].values.tolist()
#x4=df4['Review'].values.tolist()
#x = x1 + x2 + x3 + x4 
file = open('tok_corp','r')
tok_corp = file.read()
tok_corp = re.sub("[^\w]", " ",  tok_corp).split()
#tok_corp = [nltk.word_tokenize(sent) for sent in x]
model = gensim.models.Word2Vec(tok_corp, min_count=1)
print("say something")
x = input()
filtered_sentence= re.sub("[^\w]", " ",  x).split()
# print(filtered_sentence)
# x = raw_input()
returned_words.append(model.predict_output_word(filtered_sentence))
print(returned_words)

# with open('parrot.pkl', 'a') as f:
# 	pickle.dump(returned_words, f)

# print(returned_words)
# print (x)
