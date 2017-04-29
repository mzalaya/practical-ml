'''Building a Regressor Ridge'''

import sys
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics as sm

filename = sys.argv[1]
X = []
y = []

with open(filename ,'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[: -1], data[-1]
        X.append(xt)
        y.append(yt)

# Train / Test data
num_training = int(0.8*len(X))
num_test = len(X) - num_training

# Training data
#X_train = np.array(X[:num_training]).reshape((num_training, 1))
X_train = np.array(X[:num_training])
y_train = np.array(y[: num_training])

# Test data
#X_test = np.array(X[num_training: ]).reshape((num_test, 1))
X_test = np.array(X[num_training: ])
y_test = np.array(y[num_training: ])

# Create a Linear Regression Object
linear_regressor = linear_model.LinearRegression()
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

# Train the model using the training data sets
linear_regressor.fit(X_train, y_train)
ridge_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)
y_test_pred_ridge = ridge_regressor.predict(X_test)

# Measure Performance
print('LINEAR:')
print('Mean Absolute Error = ', round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print('Mean Squared Error = ', round(sm.mean_squared_error(y_test, y_test_pred), 2))
print('Mediam Absolute Error = ', round(sm.median_absolute_error(y_test, y_test_pred), 2))
print('Explain Variance Score = ', round(sm.explained_variance_score(y_test, y_test_pred), 2))
print('R2 Score = ', round(sm.r2_score(y_test, y_test_pred), 2))

print('\nRIDGE')
print('Mean Absolute Error = ', round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print('Mean Squared Error = ', round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print('Mediam Absolute Error = ', round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print('Explain Variance Score = ', round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print('R2 Score = ', round(sm.r2_score(y_test, y_test_pred), 2))

'''Building Polynomial Regressor'''
# Polynomial Regression
polynomial = PolynomialFeatures(degree=3)
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = [0.39, 2.78, 7.11]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train, y_train)

#print('Shape of X', np.array(X).shape)
#print('type of X', type(X))
print('\nLinear Regression:\n', linear_regressor.predict(datapoint))
print('\nPolynomial Regression:\n', poly_linear_model.predict(poly_datapoint))

