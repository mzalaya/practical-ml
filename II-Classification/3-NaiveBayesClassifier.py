'''Building a Naive Bayes Classifier'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

from LogisticRegression import plot_classifier
#from LogisticRegression import plot_classifier

#input_file = './Regression/data_multivar.txt'
input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[: -1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# Compute the Accuracy of the Classifier
'''
accuracy = 100.0 * (y==y_pred).sum() / X.shape[0]
print('Accuracy of the classifier =', round(accuracy, 2), '%')
plot_classifier(classifier_gaussiannb, X, y)
'''

'''Splitting the Dataset for Training and Testing - Video13'''

# Train / Test Data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb_new.predict(X_test)

'''
# Compute Acuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print('Accuracy of the classifier =', round(accuracy, 2), '%')

plot_classifier(classifier_gaussiannb_new, X_test, y_test)
'''

'''Evaluating the Accuracy Using Cross-Validation - Video 14'''

num_validations = 5
accuracy =  cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print('Accuracy: '+ str(round(100 * accuracy.mean(), 2)) + '%')

f1 = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
print('F1:: '+ str(round(100 * f1.mean(), 2)) + '%')

precision = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted',
                                             cv=num_validations)
print('Precision: '+ str(round(100 * precision.mean(), 2)) + '%')

recall = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted',
                                             cv=num_validations)
print('Recall: '+ str(round(100 * precision.mean(), 2)) + '%')

