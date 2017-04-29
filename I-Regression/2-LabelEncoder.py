
import numpy as np
from sklearn import preprocessing

'''Label Encoding'''

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'bmw']
label_encoder.fit(input_classes)

# Print Classes
print('\nClass mapping:')
for i, item in enumerate(label_encoder.classes_):
    print(item, i)

# Transform a set of classes
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print('\nLabels =', labels)
print('\nCoded Labels = ', list(encoded_labels))

# Inverse Transform
encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('\nEncoded Labels Bis =', encoded_labels)
print('Decoded Labels Bis = ', list(decoded_labels))

