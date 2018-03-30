#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:31:29 2018

@author: jaideeprao
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords

df1=pd.read_csv('movie_reviews_data/test-pos.csv'); 
df2=pd.read_csv('movie_reviews_data/test-neg.csv');
df3=pd.read_csv('movie_reviews_data/train-pos.csv');
df4=pd.read_csv('movie_reviews_data/train-neg.csv');

stop_words = set(stopwords.words('english'))
stop_words.add("I")
x1=df1['Reviews'].values.tolist()
x2=df2['Review'].values.tolist()
x3=df3['Review'].values.tolist()
x4=df4['Review'].values.tolist()
x = x1 + x2 + x3 + x4 
tok_corp = [nltk.word_tokenize(sent) for sent in x]
file = open('tok_corp','w')
file.write(str(tok_corp))
file.close()