import numpy  as np
import matplotlib.pyplot as plt

import utilities

# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

# Separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

'''
# Plot the input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolors='red', edgecolors='red', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolors='blue', edgecolors='blue', marker='s')
plt.title('Input Data')
plt.show()
'''

# Train / Test split and SVM training
from sklearn import cross_validation
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

# Linear SVM Classifier
#params = {'kernel': 'linear'}

# Building Nonlinear Classifier Using SVMs

# Using Polynomial function
# params = {'kernel': 'poly', 'degree': 3}

# Using Radial Basis funciton
params = {'kernel': 'rbf'}

classifier = SVC(**params)
classifier.fit(X_train, y_train)
'''utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')'''

y_test_pred = classifier.predict(X_test)
classifier.fit(X_test, y_test)
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset')

# Evaluate classifiers performances
from sklearn.metrics import classification_report

target_names = ['Class-' + str(int(i)) for i in set(y)]
print('#'*30 + '\n')
print('\nClassifier Performance on Training Dataset\n')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#'*30 + '\n')

print('#'*30 + '\n')
print('\nClassifier Performance on Test Dataset\n')
print(classification_report(y_test, classifier.predict(X_test), target_names=target_names))
print('#'*30 + '\n')