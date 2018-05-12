import fasttext
import settings
import os

Main_Path = os.path.join(settings.default_path)
os.chdir(Main_Path)

# Skipgram model
cbow = fasttext.cbow(os.path.join(Main_Path, 'data/movie_reviews_data/consolidated.txt'), os.path.join(Main_Path, 'clusterGenerator/fasttext_models/fast_movie_reviews/fast_model'))
# print (cbow.words) # list of words in dictionary
