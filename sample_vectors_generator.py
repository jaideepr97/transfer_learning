import fasttext

# Skipgram model
cbow = fasttext.cbow('test-pos.txt', 'sample_vectors')
print (cbow.words) # list of words in dictionary

