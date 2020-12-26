import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Loading data
pickle_in = open("training_data/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("training_data/Y.pickle", "rb")
Y = pickle.load(pickle_in)

X = X/255.0

# Creating model
model = Sequential()
# First layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# Second layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# Third layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
optimizer="adam",
metrics=['accuracy'])

model.fit(X,Y, batch_size=64,epochs=10, validation_split=0.1)