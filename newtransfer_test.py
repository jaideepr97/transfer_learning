import numpy 
import tensorflow as tf 
#from newtransfer_test import dictionary
from newtransfer_pretrain import get_dictionary
#from sklearn.cross_validation import train_test_split
import pandas as pd

def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def classify():
    global x_sent, y_sent, X, y,v
    
    #dataset4 = pd.read_csv("unassigned_water.csv")
    #dictionary = get_dictionary()
    #lvl0_x = dataset4.iloc[0:14, 0].values.tolist()
    #lvl1_x = dataset4.iloc[15:24, 0].values.tolist()
    #lvl2_x = dataset4.iloc[25:32, 0].values.tolist()
    
    dataset4 = pd.read_csv("our_data/heights1.csv")
    dictionary = get_dictionary()
    lvl0_x = dataset4.iloc[0:15, 0].values.tolist()
    lvl1_x = dataset4.iloc[16:26, 0].values.tolist()
    lvl2_x = dataset4.iloc[27:36, 0].values.tolist()
    
    # lvl0_y = dataset4.iloc[0:15, 1].values.tolist()
   # lvl1_y = dataset4.iloc[16:26, 1].values.tolist()
    #lvl2_y = dataset4.iloc[27:36, 1].values.tolist()

    embedding_size = 4
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('word_embeddings_from_our_data/word_embeddings_from_our_data.meta')
    new_saver.restore(sess, 'word_embeddings_from_our_data/word_embeddings_from_our_data')
    all_vars = tf.get_collection('vars')
    #new_saver = tf.train.import_meta_graph('test_model.meta')
    #new_saver.restore(sess, 'test_model')
    #all_vars = tf.get_collection('vars')

    for v in all_vars:
        v = sess.run(v)
      
    #    print(v)


    
    x_sent = numpy.zeros(( ( len(lvl0_x)+len(lvl1_x)+len(lvl2_x) ), embedding_size), dtype=numpy.float)
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

    #train_X, test_X, train_y, test_y = train_test_split(x_sent, y_sent, test_size=1.0, random_state=42)

    x_size = embedding_size
    y_size = 3
    h_size = 4

    #graph1 = tf.reset_default_graph()
    
    
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))
    #updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    #sess = tf.InteractiveSession()
    
    new_saver = tf.train.import_meta_graph('trainedmodeltemp20.meta')
    with tf.Session() as sess:
        
        
        #w_1 = sess.run('w_1:0')
        #w_2 = sess.run('w_2:0')
        
        new_saver.restore(sess,"trainedmodeltemp20")
       
        
        
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])
        
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
        #init = tf.global_variables_initializer()
        #sess.run(init)
        
        #print(Variable_2)
        #print(Variable_1)
        #print("w1 is:\n")
        #print(w_1)
        
        # w_1 = tf.Variable(tf.random_normal((x_size, h_size), stddev=0.1))
        #w_2 = tf.Variable(tf.random_normal((h_size, y_size), stddev=0.1))
        
        
        for epoch in range(1500):
            test_accuracy  = numpy.mean(numpy.argmax(y_sent, axis=1) == sess.run(predict, feed_dict={X: x_sent, y: y_sent}))
            print("Epoch = %d, test accuracy = %.2f%%" % (epoch + 1, 100. * test_accuracy))
            #print(sess.run(predict, feed_dict={X: x_sent, y: y_sent}))
        
        
        
        #print(sess.run(predict, feed_dict={X: x_sent, y: y_sent}))
        #X = graph1.get_tensor_by_name("XX:0")
        #y = graph1.get_tensor_by_name("yy:0")
        #print(prediction)
        #print(X)
        #for epoch in range(1500):
        #for i in range(len(train_X)):
        #    sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        #train_accuracy = numpy.mean(numpy.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
        #    test_accuracy  = numpy.mean(numpy.argmax(y_sent, axis=1) == sess.run(predictor, feed_dict={X: x_sent, y: y_sent}))
        #print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        #print("Epoch = %d, test accuracy = %.2f%%" % (epoch + 1, 100. * test_accuracy))
        
    sess.close()

classify()
