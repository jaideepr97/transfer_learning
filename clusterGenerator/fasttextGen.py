import fasttext
import re
import os
from gensim.models.wrappers import FastText
import settings

Main_Path = os.path.join(settings.default_path, 'clusterGenerator')
print(Main_Path)
os.chdir(Main_Path)
model = FastText.load_fasttext_format('sample_vectors.bin')

filtered_sentence = []
returned_words = []

def return_list(sent):
	x = sent
	filtered_sentence= re.sub("[^\w]", " ",  x).split()

	for word in filtered_sentence:
		returned_words.append(model.most_similar(word))
	return returned_words
