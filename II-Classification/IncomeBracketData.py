import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

input_file = 'adult.data'

# Redaing the data
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue

        data = line[: -1].split(', ')
        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k = 1 + count_lessthan50k
        elif data[-1] == '>50K' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k = 1 + count_morethan50k

        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break
X = np.array(X)
print(X.shape)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y= X_encoded[:, -1].astype(int)

# Build a classifier
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)

# Cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb =GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)

# Compute F1 score of the classifier
f1 = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=5)
print('F1 score: ' + str(round(100 * f1.mean(), 2)) + '%')

# Testing encoding of the classifier
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical',
              'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count = 0
#label_encoder = []
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        #label_encoder.append(preprocessing.LabelEncoder())
        input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
        count += count

input_data_encoded = np.array(input_data_encoded)

# Predict and print output for a particular datapoint
#output_class = classifier_gaussiannb.predict(input_data_encoded)
#print(label_encoder[-1].inverse_transform(output_class)[0])

'''Video<=19'''

