import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file_1 = 'cifar-10-batches-py\data_batch_1'
datafile_1 = unpickle(file_1)

X = datafile_1[b'data']
y = datafile_1[b'labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Too much load on the ram 

'''file_2 = 'cifar-10-batches-py\data_batch_2'
datafile_2 = unpickle(file_2)
file_3 = 'cifar-10-batches-py\data_batch_3'
datafile_3 = unpickle(file_3)
file_4 = 'cifar-10-batches-py\data_batch_4'
datafile_4 = unpickle(file_4)
file_5 = 'cifar-10-batches-py\data_batch_5'
datafile_5 = unpickle(file_5)

test_file = 'cifar-10-batches-py/test_batch'
testfile = unpickle(test_file)

X_train = np.concatenate([
    datafile_1[b'data'],
    datafile_2[b'data'],
    datafile_3[b'data'],
    datafile_4[b'data'],
    datafile_5[b'data']
], axis=0)

y_train = np.concatenate([
    datafile_1[b'labels'],
    datafile_2[b'labels'],
    datafile_3[b'labels'],
    datafile_4[b'labels'],
    datafile_5[b'labels']
], axis=0)

X_test = testfile[b'data']
y_test = np.array(testfile[b'labels'])'''

# reshape(-1, 3, 32, 32) -> (batch_size, RGB, height, width)
# transpose(0, 2, 3, 1) -> to change RGB channels to last -> tensorflow/keras image format
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Normalizing to float32 in range[0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Resizing images as VGG takes input size of (224, 224)
X_train = tf.image.resize(X_train, [224, 224]).numpy()
X_test = tf.image.resize(X_test, [224, 224]).numpy()

plt.figure(figsize=(15,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i])
plt.tight_layout()
plt.show()

# Saving the preprocessed data
np.savez_compressed('cifar10_preprocessed.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)