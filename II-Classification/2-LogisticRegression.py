'''Building a Logistic Regression Classifier'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def plot_classifier(classifier, X, y):

    # Define ranges to plot the figure
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # Denotes the step size that will be used in the mesh grid
    step_size = 0.1

    # Define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # Compute the Classifier Output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # Reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # Choose a color schame you can find all the options
    # here: http://matplotlib.org/examples/colorsreferences/html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.Set1)

    # Overlay the Training Points of the Figure
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', linewidth=2, cmap=plt.cm.Paired)

    # Specify the Boundaries og the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()

if __name__ == '__main__':
    # input data
    X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # Initialize the logistic Regression Classifier
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100.0)

    # Train the Classifier
    classifier.fit(X, y)

    # Draw datapoints and boundaries
    plot_classifier(classifier, X, y)
