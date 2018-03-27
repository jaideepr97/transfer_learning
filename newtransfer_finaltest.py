#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:33:04 2018

@author: jaideeprao
"""

import numpy 
import tensorflow as tf 
from newtransfer_pretrain import get_dictionary
import pandas as pd

def classify(sentence):
     global x_sent, y_sent, X, y,v,dictionary
     dictionary = get_dictionary()
      
     embedding_size = 4
     sess = tf.Session()
     new_saver = tf.train.import_meta_graph('word_embeddings_from_our_data/word_embeddings_from_our_data.meta')
     new_saver.restore(sess, 'word_embeddings_from_our_data/word_embeddings_from_our_data')
     all_vars = tf.get_collection('vars')
     for v in all_vars:
        v = sess.run(v)
        
     print(sentence.split())
     x_sent = numpy.zeros((1, embedding_size), dtype=numpy.float)
     #x_sent = numpy.average( [v[dictionary[word]] for word in user_sentence.split() if word in dictionary], axis=0)
     #x_sent = numpy.transpose(x_sent)
     x_size = embedding_size
     y_size = 3
     h_size = 4
     new_saver = tf.train.import_meta_graph('trainedmodeltemp20.meta')
     with tf.Session() as sess:
         new_saver.restore(sess,"trainedmodeltemp20")
         X = tf.placeholder("float", shape=[None, x_size])
         #y = tf.placeholder("float", shape=[None, y_size])
        
         w_1 = tf.placeholder("float", shape=[x_size,h_size])
         w_2 = tf.placeholder("float", shape=[h_size,y_size])
         init = tf.global_variables_initializer()
         sess.run(init)
        
         w_1 = sess.run('w_1:0')
         w_2 = sess.run('w_2:0')
         Variable_2 = sess.run('Variable_2:0')
         Variable_1 = sess.run('Variable_1:0')
         Variable = sess.run('Variable:0')
        
         h = tf.nn.sigmoid(tf.matmul(X, w_1))
         yhat = tf.matmul(h, w_2)
         predict = tf.argmax(yhat, dimension=1)
         
         classified = sess.run(predict, feed_dict={X:x_sent})
         #print(classified)
         return classified

     sess.close()
     
#print("enter your sentence now")
#user_sentence = input()
#user_sentence = "Be thrilled and say yes"
#classify(user_sentence)     
             
         
