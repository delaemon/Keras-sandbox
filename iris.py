from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
targets = iris.target

model = Sequential()
model.add(Dense(12, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(3, input_dim=12))
model.add(Activation('softmax'))
#model.add(Activation('relu')) #Bad
model.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(features, targets, nb_epoch = 20, batch_size = 5)
