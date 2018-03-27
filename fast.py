import fasttext
import re
from gensim.models.wrappers import FastText

# Skipgram model

model = FastText.load_fasttext_format('sample_vectors.bin')

filtered_sentence = []
returned_words = []

def return_list(sent):
	x = sent
	filtered_sentence= re.sub("[^\w]", " ",  x).split()

	for word in filtered_sentence:
		returned_words.append(model.most_similar(word))

	return returned_words


#model = fasttext.load_model('sample_vectors.bin')



#cbow = fasttext.cbow('test-pos.txt', 'sample_vectors')
#print(len(cbow.words))
#print (cbow.words) # list of words in dictionary


#print(model.most_similar['incisive'])

# Output = [('headteacher', 0.8075869083404541), ('schoolteacher', 0.7955552339553833), ('teachers', 0.733420729637146), ('teaches', 0.6839243173599243), ('meacher', 0.6825737357139587), ('teach', 0.6285147070884705), ('taught', 0.6244685649871826), ('teaching', 0.6199781894683838), ('schoolmaster', 0.6037642955780029), ('lessons', 0.5812176465988159)]


#print(model.similarity('teacher', 'teaches'))
# Output = 0.683924396754
