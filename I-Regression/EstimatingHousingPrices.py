
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

def plot_features_importances(feature_importances, title, features_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the Bar Graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, features_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

if __name__ =='__main__':
    # Load Housing data
    housing_data = datasets.load_boston()

    # Shuffle the data
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.8*len(X))
    X_train, y_train = X[: num_training], y[: num_training]
    X_test, y_test = X[num_training: ], y[num_training: ]

    # Fit Decision Tree Regression Model
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # Fit Decision Regression with AdaBoost
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    # Evaluate Performance of Decision Tree Regressor
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt)
    print('\n #### Decision Tree Performance ####')
    print('Mean Squared Error = ', round(mse, 2))
    print('explained Variance Score= ', round(evs, 2))

    # Evaluate Performance of AdaBoost
    y_pred_ab = ab_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab)
    print('\n #### AdaBoost ####')
    print('Mean Squared Error = ', round(mse, 2))
    print('explained Variance Score= ', round(evs, 2))

    print('hous', housing_data.feature_names)

    # Computing Relative Importance of Features
    plot_features_importances(dt_regressor.feature_importances_, 'Decision Tree Regressor', housing_data.feature_names)
    plot_features_importances(ab_regressor.feature_importances_, 'AdaBoost Regressor', housing_data.feature_names)















