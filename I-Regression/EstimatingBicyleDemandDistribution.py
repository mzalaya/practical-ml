import csv
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

from Regression.EstimatingHousingPrices import plot_features_importances


def load_dataset(filename):
    file_reader = csv.reader(open(filename,'r'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:14])
        #X.append(row[2:15])
        #X.append(row[2:13])
        y.append(row[-1])

    # Extract features names
    feature_names = np.array(X[0])

    # Remove the first row because they are features names
    return np.array(X[1: ]).astype(np.float32), np.array(y[1: ]).astype(np.float32), feature_names

if __name__ == '__main__':
    # Load the dataset from the input file
    X, y, feature_names = load_dataset(sys.argv[1])
    X, y = shuffle(X, y, random_state=7)

    # Split the data 80/20 (80% for traininig , 20% for testing)
    num_training = int(0.9*len(X))
    X_train, y_train = X[: num_training], y[: num_training]
    X_test, y_test = X[num_training: ], y[num_training: ]

    # Fit the Random Forest Regression Model
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=1)
    rf_regressor.fit(X_train, y_train)

    # Evaluate Performance of Random Forest Regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test , y_pred)
    print('\n #### Random Forest Performance ####')
    print('Mean Squared Error = ', round(mse, 2))
    print('explained Variance Score= ', round(evs, 2))

    # Plot Relative Feature Importance
    plot_features_importances(rf_regressor.feature_importances_, 'Random Forest Regressor', feature_names)



