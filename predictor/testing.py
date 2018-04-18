import tensorflow as tf
import numpy
import os
import pickle
import settings
import pandas as pd

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)

def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def predict():
    global dataset_test, v, dictionary, x_sent, y_sent, reviews, labels
    dataset = pd.read_csv("yelp_labelled.csv",  delimiter="\t")
    with open("mySavedDict_movie_reviews_data.txt", "rb") as myFile:
        dictionary = pickle.load(myFile)
    dataset_test = dataset.iloc[400:450, 0:2]
    reviews = dataset_test.iloc[:, 0].values.tolist()
    labels = dataset_test.iloc[:, 1].values.tolist()

    embedding_size = 4
    no_of_labels = 2

    with open("word_embeddings_from_movie_reviews_data/word_embeddings_from_movie_reviews_data.txt", "rb") as myFile:
        v = pickle.load(myFile)
        v = numpy.asarray(v, dtype=float)
    x_sent = numpy.zeros(((len(dataset_test)), embedding_size), dtype=numpy.float)
    y_sent = numpy.zeros(((len(dataset_test)), no_of_labels), dtype=numpy.float)

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


    Main_Path = os.path.join(settings.default_path, 'predictor')
    os.chdir(Main_Path)

    with tf.Session() as sess:
    #  new_saver = tf.train.import_meta_graph('testsaver.meta')

      new_saver = tf.train.import_meta_graph('tensnet/trainedmodel(17-4-18).meta')
      new_saver.restore(sess, 'tensnet/trainedmodel(17-4-18)')
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


predict()

#  test_accuracy  = numpy.mean(numpy.argmax(y_sent, axis=1) == sess.run(predict, feed_dict={x: x_sent, y: y_sent}))
#  print(test_accuracy)
#  writer = tf.summary.FileWriter('./log/', sess1.graph)
#  writer.close()
  #print(b2)
