import tensorflow as tf
import numpy
import pickle
import pandas as pd
#from tensnet import getV


def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

dataset4 = pd.read_csv("our_data/heights1.csv")
with open("mySavedDict_movie_reviews_data.txt", "rb") as myFile:
    dictionary = pickle.load(myFile)
lvl0_x = dataset4.iloc[0:15, 0].values.tolist()
lvl1_x = dataset4.iloc[16:26, 0].values.tolist()
lvl2_x = dataset4.iloc[27:36, 0].values.tolist()

embedding_size = 4
#with tf.Session() as sess1:  
#    new_saver = tf.train.import_meta_graph('word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.meta')
#    new_saver.restore(sess1, 'word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data')
#    graph = tf.get_default_graph()
#    all_vars = tf.get_collection('vars')
#    for v in all_vars:
#        v = sess1.run(v)
#    writer = tf.summary.FileWriter('./log/', sess1.graph)
#    writer.close()    
#sess1.close()   
#v = getV()  
with open("word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.txt", "rb") as myFile:
    v = pickle.load(myFile)
    v = numpy.asarray(v, dtype=float)
x_sent = numpy.zeros(( ( len(lvl0_x)+len(lvl1_x)+len(lvl2_x) ), embedding_size), dtype=numpy.float)
y_sent = numpy.zeros(((len(lvl0_x) + len(lvl1_x) + len(lvl2_x)), 3), dtype=numpy.float)

for i, sent in enumerate(lvl0_x):
    y_sent[i, 0] = 1
    x_sent[i] = numpy.average([v[dictionary[word]] for word in sent.split() if word in dictionary], axis=0)

for i, sent in enumerate(lvl1_x):
    y_sent[len(lvl0_x) + i, 1] = 1
    x_sent[len(lvl0_x) + i] = numpy.average([v[dictionary[word]] for word in sent.split() if word in dictionary], axis=0)

for i, sent in enumerate(lvl2_x):
    y_sent[len(lvl0_x) + len(lvl1_x) + i, 2] = 1
    x_sent[len(lvl0_x) + len(lvl1_x) + i] = numpy.average([v[dictionary[word]] for word in sent.split() if word in dictionary], axis=0)

col_mean = numpy.nanmean(x_sent, axis=0)
inds = numpy.where(numpy.isnan(x_sent))    
x_sent[inds] = numpy.take(col_mean, inds[1])
x_sent, y_sent = shuffle_in_unison_scary(x_sent, y_sent)
#
#
#print(sess1._closed)

#x_sent = numpy.zeros(( 34, 4), dtype=numpy.float)
#y_sent = numpy.zeros((34, 3), dtype=numpy.float)

with tf.Session() as sess:
#  new_saver = tf.train.import_meta_graph('testsaver.meta')

  new_saver = tf.train.import_meta_graph('tensnet/trainedmodel(7-4-18).meta')
  new_saver.restore(sess, 'tensnet/trainedmodel(7-4-18)')
#  new_saver = tf.train.import_meta_graph('tensnet/best_model_yet(81.82)/trainedmodel(7-4-18).meta')
#  new_saver.restore(sess, 'tensnet/best_model_yet(81.82)/trainedmodel(7-4-18)')
  
  
  graph = tf.get_default_graph()
  
  
#  x = graph.get_tensor_by_name("x") 
#   now declare the output data placeholder - 10 digits
#  y = graph.get_tensor_by_name("y") 
  
#  x = sess.graph.get_tensor_by_name('Placeholder:0')
  x = sess.graph.get_tensor_by_name('x:0')  
#   now declare the output data placeholder - 10 digits
#  y = sess.graph.get_tensor_by_name('Placeholder_1:0')
  y = sess.graph.get_tensor_by_name('y:0')  

  W1 = graph.get_tensor_by_name("W1:0") 
  W2 = graph.get_tensor_by_name("W2:0")
  b1 = graph.get_tensor_by_name("b1:0") 
  b2 = graph.get_tensor_by_name("b2:0")  
#  feed_dict ={x: mnist.test.images, y: mnist.test.labels}
  
#  hidden_out = graph.get_tensor_by_name("hidden_out:0")

  h = graph.get_tensor_by_name("h:0")

#  y_ = graph.get_tensor_by_name("y_:0")
  
  yhat = graph.get_tensor_by_name("yhat:0")  
  correct_prediction = graph.get_tensor_by_name("correct_prediction:0")
  accuracy = graph.get_tensor_by_name("accuracy:0")
#  predict = tf.argmax(yhat, dimension=1, name="predict")

  print(sess.run(W1))
  print(sess.run(W2))
  print(sess.run(b1))
  print(sess.run(b2))
  print(sess.run(accuracy, feed_dict={x:x_sent, y: y_sent})) 
#  test_accuracy  = numpy.mean(numpy.argmax(y_sent, axis=1) == sess.run(predict, feed_dict={x: x_sent, y: y_sent}))
#  print(test_accuracy)
#  writer = tf.summary.FileWriter('./log/', sess1.graph)
#  writer.close()
  #print(b2)
