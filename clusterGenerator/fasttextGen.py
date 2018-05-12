import fasttext
import re
import os
from gensim.models.wrappers import FastText
import settings

Main_Path = os.path.join(settings.default_path, 'clusterGenerator/fasttext_models/fast_movie_reviews')
os.chdir(Main_Path)

# print("Loading fasttext model")
model = FastText.load_fasttext_format('fast_model.bin')
# print("Loaded")

def return_list(sent):
	returned_words = []
	filtered_sentence = re.sub("[^\w]", " ",  sent).split()
	for word in filtered_sentence:
		try:
			returned_words.append(model.most_similar(word))
		except KeyError:
			continue
	return returned_words
