
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_openml
from utils import *
import pickle as pkl
import numpy as np
import pdb

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D

import keras.layers as kl

#made a pr test1

#load data
#mnist=fetch_openml('mnist_784')
#pdb.set_trace()

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
mnistm=pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

#Preprocess
mnist_train=(mnist.train.images>0).reshape(55000,28,28,1).astype(np.uint8) * 255
#mnist_train=np.concatenate([mnist_train,mnist_train,mnist_train],3)
mnist_test=(mnist.test.images>0).reshape(10000,28,28,1).astype(np.uint8) * 255
#mnist_test=np.concatenate([mnist_test,mnist_test,mnist_test],3)

num_test = 500
#combined_test_imgs=np.vstack([mnist_test[:num_test],mnistm_test[:num_test]])
#combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
#combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),np.tile([0., 1.], [num_test, 1])])

#imshow_grid(mnist_train)
#imshow_grid(mnistm_train)

model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(mnist_train, mnist.train.labels, validation_data=(mnist_test, mnist.test.labels), epochs=3)
model.predict(mnist_test[:4])
pdb.set_trace()






class DANN_Keras_model(object):
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))

    model.add(Conv2D(48, kernel_size=3, activation='relu'))

    model.add(Flatten())
    #model.add(Dense(10, activation='softmax'))
    def _init_(self,features=10):
        self.features_dim = features

    def feature_extactor(self,input):
        out = kl.Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(input)
        out = kl.MaxPool2D(pool_size=(2,2))(out)

        out=kl.Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(out)
        out = kl.MaxPool2D(pool_size=(2, 2))(out)

        features = kl.Dense(self.features_dim, activation="relu")(out)
        self.domain_invariant_features = features

        return features

    def label_predictor(self,input):
        out = kl.Dense(100, activation="relu")(input)
        out = kl.MaxPool2D(pool_size=(2,2))(out)

        out=kl.Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(out)
        out = kl.MaxPool2D(pool_size=(2, 2))(out)

        features = kl.Dense(self.features_dim, activation="relu")(out)
        self.domain_invariant_features = features

        return features







class DANN_tf_model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X= tf.placeholder(tf.uint8,[None,28,28,3])
        self.y= tf.placeholder(tf.float,[None,10])
        #self.=

        X_input=tf.cast(self.X)
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 7 * 7 * 48])