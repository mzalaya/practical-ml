import numpy as np
import matplotlib.pyplot as plt

# Input Data
X = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])

# Labels
y = [0, 1, 1, 0, 0, 1, 1, 0]

# Separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

# Plot input data
'''
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], color='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], color='black', marker='x')
plt.show()
'''
# Draw the Separator Line
line_x = range(10)
line_y = line_x

# Plot Labeled Data and Seperator Line
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], color='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], color='black', marker='x')
plt.plot(line_x, line_y, color='black', linewidth=3)
plt.show()