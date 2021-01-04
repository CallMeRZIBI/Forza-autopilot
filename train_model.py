import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

# Loading data
data = np.load("training_data/data.npz")
X = data['arr_0']

Y = data['arr_1']

X = X/255.0

#--------------Optimalization-------------
dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

# Creating model
model = Sequential()

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            Name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size,dense_layer, int(time.time()))
            print(Name)
            # TensorBoard
            tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

            # First layer
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                # Second layer
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            # Third layer
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            model.add(Dense(4))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

            model.fit(X,Y, batch_size=64,epochs=20, validation_split=0.1, callbacks=[tensorboard])

model.save("model/{}x{}x{}-CNN.model".format(layer_size,conv_layer,dense_layer))