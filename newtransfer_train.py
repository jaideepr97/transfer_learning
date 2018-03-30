import numpy 
import tensorflow as tf 
import pickle
#from newtransfer_test import dictionary
#from newtransfer_pretrain import get_dictionary
from sklearn.cross_validation import train_test_split
import pandas as pd
a = 5
def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def classify():
    global x_sent, y_sent, train_X, test_X, train_y, test_y, dictionary, v, k, test_accuracy 
    dataset4 = pd.read_csv("our_data/heights1.csv")
    #   dictionary = get_dictionary()
    lvl0_x = dataset4.iloc[0:15, 0].values.tolist()
    lvl1_x = dataset4.iloc[16:26, 0].values.tolist()
    lvl2_x = dataset4.iloc[27:36, 0].values.tolist()

   # lvl0_y = dataset4.iloc[0:15, 1].values.tolist()
   # lvl1_y = dataset4.iloc[16:26, 1].values.tolist()
    #lvl2_y = dataset4.iloc[27:36, 1].values.tolist()

    embedding_size = 4
    sess = tf.Session()
    #new_saver = tf.train.import_meta_graph('word_embeddings_from_our_data/word_embeddings_from_our_data.meta')
    #new_saver.restore(sess, 'word_embeddings_from_our_data/word_embeddings_from_our_data')
    
    new_saver = tf.train.import_meta_graph('word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.meta')
    new_saver.restore(sess, 'word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data')
    
    
    all_vars = tf.get_collection('vars')
    with open("mySavedDict_movie_reviews_data.txt", "rb") as myFile:
        dictionary = pickle.load(myFile)
    #new_saver = tf.train.import_meta_graph('test_model.meta')
    #new_saver.restore(sess, 'test_model')
    #all_vars = tf.get_collection('vars')

    for v in all_vars:
        v = sess.run(v)
      
    #    print(v)



    x_sent = numpy.zeros(((len(lvl0_x) + len(lvl1_x) + len(lvl2_x)), embedding_size), dtype=numpy.float)
    # #y_sent = numpy.zeros((len(lvl0_y + lvl1_y + lvl2_y), 3), dtype=numpy.int)
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

    train_X, test_X, train_y, test_y = train_test_split(x_sent, y_sent, test_size=0.3, random_state=42)

    x_size = embedding_size
    y_size = 3
    h_size = 4


    X = tf.placeholder("float", shape=[None, x_size], name="X")
    y = tf.placeholder("float", shape=[None, y_size], name="y")
    w_1 = tf.Variable(tf.random_normal((x_size, h_size), stddev=0.1), name="w_1")
    w_2 = tf.Variable(tf.random_normal((h_size, y_size), stddev=0.1) , name="w_2")
    h = tf.nn.sigmoid(tf.matmul(X, w_1), name="h")
    yhat = tf.matmul(h, w_2)
    predict = tf.argmax(yhat, dimension=1, name="predict")
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    file = open("current_highest_test_accuracy.txt","r") 
    k = file.read()
    k = float(k)
    file.close()
    

    for epoch in range(1500):
         
         for i in range(len(train_X)):
             sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
         train_accuracy = numpy.mean(numpy.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
         test_accuracy  = numpy.mean(numpy.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
         print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
         #print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))
         if(test_accuracy > k):
             k = test_accuracy
             file = open("current_highest_test_accuracy.txt","w")
             file.write(str(k))
             file.close()
             saver = tf.train.Saver({"w_1":w_1, "w_2":w_2})
             saver.save(sess, "/home/jaideeprao/Desktop/transfer_learning/trainedmodel(29-3-18)")
     
     

    
    
    #saver = tf.train.Saver({"w_1":w_1, "w_2":w_2})
    #saver.save(sess, "/home/jaideeprao/Desktop/transfer_learning/trainedmodel(29-3-18)")
    
    #w_1 = sess.run('w_1:0')
    #w_2 = sess.run('w_2:0')
    #print(w_1, w_2)
    #print(X.eval())
    sess.close()
    

classify()
