import pandas
import os
import settings
from clusterGenerator.clusterController import generateCluster

Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)
resulting_input_cluster = []


print("Loading test data")
input_data = pandas.read_csv('yelp_labelled_1.csv', delimiter=",", header=None)
print("Loaded")

input_data_length = input_data.shape[0]
for sentence in input_data.itertuples():
    # print(sentence)
    resulting_input_cluster.append(generateCluster(sentence._1, sentence.Index))

# print(resulting_input_cluster)
