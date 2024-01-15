import pandas as pd
import numpy as np
from dtaidistance import dtw
from sklearn.neighbors import KNeighborsClassifier

#Read in both data sets
yahoo = pd.read_csv("yahoo.csv", sep=',', header=0)
bloomberg = pd.read_csv("bloomberg.csv", sep=',', header=0)
print("CSVs loaded!")

bloomberg_list = []
yahoo_list = []
temp_list = []
#Iterate over each point in the data
for x, y in bloomberg.iterrows():
    temp_list.append(y[2])
    #If it's a multiple of 7, we know we completed and interval
    if (y[0] % 7 == 0):
        bloomberg_list.append(temp_list[0::50])
        print(len(bloomberg_list), " points in Bloomberg")
        temp_list.clear()
temp_list.clear()
print("Bloomberg Done!")
for x, y in yahoo.iterrows():
    temp_list.append(y[2])
    if (y[0] % 7 == 0):
        yahoo_list.append(temp_list[0::50])
        print(len(yahoo_list), " points in Yahoo")
        temp_list.clear()
print("Yahoo Done!")

print(len(bloomberg_list))
print(len(yahoo_list))

bloom_distances = []
yahoo_distances = []
#DTW Analysis compares each interval to the following ones.
for x in range(1, len(bloomberg_list)):
    for j in range(x+1, len(bloomberg_list)):
        bloom_distances.append(dtw.distance(bloomberg_list[x], bloomberg_list[j]))
        yahoo_distances.append(dtw.distance(yahoo_list[x], yahoo_list[j]))
        print("Currently on: ", x, " : ", j)

#Splitting data into test and train for KNN
bloom_train = bloom_distances[1:len(bloom_distances)-10]
bloom_train_labels = []
for i in range(len(bloom_train)):
    bloom_train_labels.append("Bloomberg")
bloom_test = bloom_distances[len(bloom_distances)-9:]

yahoo_train = yahoo_distances[1:len(yahoo_distances)-10]
yahoo_train_labels = []
for i in range(len(yahoo_train)):
    yahoo_train_labels.append("Yahoo")
yahoo_test = yahoo_distances[len(yahoo_distances)-9:]

train_data = bloom_train + yahoo_train
train_data_labels = bloom_train_labels + yahoo_train_labels
test_data = bloom_test + yahoo_test

train_data = list(map(lambda x:[x], train_data))
test_data = list(map(lambda x:[x], test_data))

for x in range(len(train_data_labels)):
    if train_data_labels[x] == "Bloomberg":
        train_data_labels[x] = 1
    elif train_data_labels[x] == "Yahoo":
        train_data_labels[x] = 0
#Perform KNN on training data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data, train_data_labels)
print("Fitted")
#Testing it
for i in range(len(test_data)):
    x = neigh.predict([test_data[i]])
    if (x == 1):
        print("Bloomberg")
    elif (x == 0):
        print("Yahoo")