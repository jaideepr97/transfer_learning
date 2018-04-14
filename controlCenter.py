import pandas
import numpy
import os
import settings
from clusterGenerator.clusterController import generateCluster

# print(settings.default_path)
Main_Path = os.path.join(settings.default_path, 'data')
os.chdir(Main_Path)
resulting_input_cluster = []

input_data = pandas.read_csv('input_data.csv');
input_data_length = input_data.shape[0]
for sentence in input_data.itertuples():
    resulting_input_cluster.append(generateCluster(sentence._1))

print(resulting_input_cluster)
