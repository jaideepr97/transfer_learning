import pandas
import os
import settings
# print(os.getcwd())
from clusterGenerator.clusterController import generateCluster
os.chdir(settings.default_path)
# print(os.getcwd())
from predictor.predictor import predict

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)
resulting_input_cluster = []


# print("Loading test data")
# input_data = pandas.read_csv('yelp_labelled.csv', delimiter="\t")
# trunc_input_data = input_data.iloc[40:70, 0:2]
# trunc_input_labels = trunc_input_data.iloc[:, 1].values.tolist()
# trunc_input_reviews = input_data.iloc[0:20, 0:2].values.tolist()
# print("Loaded")
#
#
# trunc_input_data_length = trunc_input_data.shape[0]
#
# for sentence in trunc_input_data.itertuples():
# # for i, sentence in enumerate(trunc_input_reviews):
#     # print(sentence)
#     resulting_input_cluster.append(generateCluster(sentence._1, sentence.Index))
#     # resulting_input_cluster.append(generateCluster(sentence, i))
# # print(resulting_input_cluster)
# predict(resulting_input_cluster   , trunc_input_labels)


print("Initializing testing")
input_data = pandas.read_csv('yelp_labelled.csv', delimiter="\t")
i = 0;
while(i<len(input_data)):
    print("Loading rows " + str(i) + " to " + str(i+20))
    trunc_input_data = input_data.iloc[i:i+20, 0:2]
    trunc_input_labels = trunc_input_data.iloc[:, 1].values.tolist()
    trunc_input_reviews = trunc_input_data.iloc[:, 0].values.tolist()
    print("Loaded")
    trunc_input_data_length = trunc_input_data.shape[0]
    resulting_input_cluster = []
    for sentence in trunc_input_data.itertuples():
    # for i, sentence in enumerate(trunc_input_reviews):
        # print(sentence)

        try:
            resulting_input_cluster.append(generateCluster(sentence._1, sentence.Index))
        except TypeError:
            continue
        # resulting_input_cluster.append(generateCluster(sentence, i))
    # print(resulting_input_cluster)
    predict(resulting_input_cluster, trunc_input_labels)
    i = i+20;
