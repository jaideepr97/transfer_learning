import numpy
import tensorflow as tf
import pickle
import os
from sklearn.cross_validation import train_test_split
import pandas as pd
import settings

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)

def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def classify():
    global x_sent, y_sent, train_X, test_X, train_y, test_y, dictionary, v, k, test_accuracy, dataset
    dataset = pd.read_csv("yelp_labelled.csv",  delimiter="\t")
    dataset_train = dataset.iloc[0:50, 0:2]
    reviews = dataset_train.iloc[:, 0].values.tolist()
    labels = dataset_train.iloc[:, 1].values.tolist()

    embedding_size = 4
    no_of_labels = 2
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.meta')
        new_saver.restore(sess, 'word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data')

        graph = tf.get_default_graph()

        all_vars = tf.get_collection('vars')
        with open("mySavedDict_movie_reviews_data.txt", "rb") as myFile:
            dictionary = pickle.load(myFile)
        for v in all_vars:
            v = sess.run(v)
            with open("word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.txt", "wb") as myFile:
                pickle.dump(v, myFile)

    x_sent = numpy.zeros(((len(dataset_train)), embedding_size), dtype=numpy.float)
    y_sent = numpy.zeros(((len(dataset_train)), no_of_labels), dtype=numpy.float)


    for i, sent in enumerate(reviews):

        if(labels[i] == 1):
            y_sent[i, 1] = 1
        else:
            y_sent[i, 0] = 1
        x_sent[i] =  numpy.average([v[dictionary[word]] for word in sent.split() if word in dictionary], axis=0)

    col_mean = numpy.nanmean(x_sent, axis=0)
    inds = numpy.where(numpy.isnan(x_sent))
    x_sent[inds] = numpy.take(col_mean, inds[1])
    x_sent, y_sent = shuffle_in_unison_scary(x_sent, y_sent)

    train_X, test_X, train_y, test_y = train_test_split(x_sent, y_sent, test_size=0.3)

    x_size = embedding_size
    y_size = 2
    h_size = 4

    learning_rate = 0.5
    epochs = 1500

    x = tf.placeholder(tf.float32, [None, x_size], name="x")
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, y_size], name="y")

    W1 = tf.Variable(tf.random_normal([x_size, h_size], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([h_size]), name='b1')

    W2 = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([y_size]), name='b2')

#    h = tf.nn.sigmoid(tf.matmul(x, W1), name="h")
    h = tf.nn.relu(tf.add(tf.matmul(x, W1),b1), name="h")

#    yhat = tf.matmul(h, W2, name="yhat")
    yhat = tf.nn.softmax(tf.add(tf.matmul(h, W2),b2), name="yhat")

    predict = tf.argmax(yhat, dimension=1, name="predict")

    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))

    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init_op = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1), name="correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    file = open("current_highest_test_accuracy.txt","r")
    k = file.read()
    k = float(k)
    file.close()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter("output", sess.graph)
        for epoch in range(epochs):
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={x: train_X[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = numpy.mean(numpy.argmax(train_y, axis=1) == sess.run(predict, feed_dict={x: train_X, y: train_y}))
            test_accuracy  = numpy.mean(numpy.argmax(test_y, axis=1) == sess.run(predict, feed_dict={x: test_X, y: test_y}))
            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            print(sess.run(predict, feed_dict={x: test_X, y: test_y}))
            if(test_accuracy > k):
                 k = test_accuracy
                 file = open("current_highest_test_accuracy.txt","w")
                 file.write(str(k))
                 file.close()
                 saver = tf.train.Saver()
                 saver.save(sess, "/home/jaideeprao/Desktop/finalStretch/predictor/tensnet/trainedmodel(17-4-18)")
        print (tf.shape(x))
        print (tf.shape(y))
        # saver = tf.train.Saver()
        # writer = tf.summary.FileWriter('./log/', sess.graph)
        print(sess.run(W1))
        print(sess.run(W2))
        print(sess.run(b1))
        print(sess.run(b2))
        # saver.save(sess, "/home/jaideeprao/Desktop/finalStretch/predictor/tensnet/trainedmodel(17-4-18)")
        writer.close()


classify()
