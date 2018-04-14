import gensim
import settings
import pandas as pd
import nltk
import re
import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
Main_Path = os.path.join(settings.default_path, 'data')

os.chdir(Main_Path)
from .fasttextGen import return_list
from .w2vGen import word2vec_words

from subprocess import Popen, PIPE

print("here bitchexzxxx")
df1=pd.read_csv('movie_reviews_data/test-pos.csv');
df2=pd.read_csv('movie_reviews_data/test-neg.csv');
df3=pd.read_csv('movie_reviews_data/train-pos.csv');
df4=pd.read_csv('movie_reviews_data/train-neg.csv');


x1=df1['Reviews'].values.tolist()
x2=df2['Review'].values.tolist()
x3=df3['Review'].values.tolist()
x4=df4['Review'].values.tolist()
x = x1 + x2 + x3 + x4
tok_corp = [nltk.word_tokenize(sent) for sent in x]
model = gensim.models.Word2Vec(tok_corp, min_count=1)

stop_words = set(stopwords.words('english'))
stop_words.add("I")
filtered_sentence = []
returned_words = []
returned_prob = []
returned = []
cmd_returned = []
new_clust = []
new_cmd_returned = []
new_sorted_obb = []
new_collected_obb = []
new_averaged_obb = []
fasttext_return = []
new_fasttext_return = []
shell_return = []
new_shell_return = []
final_list = []
present = []
maybe_present = []
w2v_returned = []

def preprocessing(testvar):
    word_tokens = word_tokenize(testvar)
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

def backend(strr, testvar):
    fasttext_return = return_list(strr)
    n = len(fasttext_return)
    i = 0
    while i < n:
        m = len(fasttext_return[i])
        j = 0
        while j < m:
            new_fasttext_return.append(fasttext_return[i][j])
            j = j + 1
        i = i + 1
    j = 1
    while j <= settings.no_of_w2v_iterations:
        cmd_returned.append(word2vec_words(testvar,j))
        j = j + 1

    new_cmd_returned = [item for sublist in cmd_returned for item in sublist]
    new_collected_cmd_returned = [(uk, sum([vv for kk,vv in new_cmd_returned if kk==uk])/5 ) for uk in set([k for k,v in new_cmd_returned])]

    new_collected_cmd_returned = sorted(new_collected_cmd_returned)


    for word in new_collected_cmd_returned:
        for element in filtered_sentence:
            if word[0] == element:
                if word[0] not in present:
                    present.append(word[0])

    for word in new_fasttext_return:
        for element in filtered_sentence:
            if word[0] == element:
                if word[0] not in present:
                    present.append(word[0])



    ft_len = len(new_fasttext_return)
    w2v_len = len(new_collected_cmd_returned)
    i = 0

    while i < w2v_len:
        j = 0
        while j < ft_len:
            if new_fasttext_return[j][1] > 0.70:
                if new_fasttext_return[j][0] not in maybe_present:
                        maybe_present.append(new_fasttext_return[j][0])
            try:
                sim = (model.wv.similarity(new_collected_cmd_returned[i][0], new_fasttext_return[j][0]))
            except KeyError:
                continue
            if sim > 0.4 :
                if new_collected_cmd_returned[i][0] not in maybe_present:
                    maybe_present.append(new_collected_cmd_returned[i][0])
            j = j + 1
        i = i + 1

    cluster = present + maybe_present
    new_clust = ' '.join(cluster)
    print("PRINTING NEW CLUST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return new_clust

def generateCluster(testvar):
    preprocessing(testvar)
    strr = " ".join(filtered_sentence)
    return backend(strr,testvar)
