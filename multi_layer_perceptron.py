#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

model = Sequential([
    Dense(512, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)
print('test accuracy : ', score[1])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(range(20), loss, marker = '.', label = 'loss')
plt.plot(range(20), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc  = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

acc = hist.history['acc']
val_acc = hist.history['val_acc']
plt.plot(range(20), acc, marker = '.', label = 'acc')
plt.plot(range(20), val_acc, marker = '.', label = 'val_acc')
plt.legend(loc  = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
