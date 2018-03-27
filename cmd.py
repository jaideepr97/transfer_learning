import gensim
import pandas as pd
import nltk
import re
import pickle

returned_words = []

df=pd.read_csv('movie_reviews_data/test-pos.csv'); 
x=df['Reviews'].values.tolist()
tok_corp = [nltk.word_tokenize(sent) for sent in x]
model = gensim.models.Word2Vec(tok_corp, min_count=1)

x = raw_input()
filtered_sentence= re.sub("[^\w]", " ",  x).split()
# print(filtered_sentence)
# x = raw_input()
returned_words.append(model.predict_output_word(filtered_sentence))


with open('parrot.pkl', 'a') as f:
	pickle.dump(returned_words, f)

# print(returned_words)
# print (x)
