import fasttext

# Skipgram model
cbow = fasttext.cbow('movie_reviews_data_text/test-pos.txt', 'sample_vectors')
print (cbow.words) # list of words in dictionary

