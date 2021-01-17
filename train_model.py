import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard
import time

# Loading data
data = np.load("training_data/data.npz",allow_pickle=True)
X = data['arr_0']

Y = data['arr_2']

Z = data['arr_1']

X = X/255.0

#--------------Optimization--------------
dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

# Creating model for image processing
model = Sequential()

# Creating model for object processing
model2 = Sequential()

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            Name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size,dense_layer, int(time.time()))
            print(Name)
            # TensorBoard
            tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

            # Model for image processing
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

            # Model for object processing
            # Inputs
            model2.add(Input(shape=(5,4)))
            # Hidden layer
            model2.add(Dense(8))
            model2.add(Activation('relu'))
            # Outputs
            model2.add(Dense(4))
            model2.add(Activation('sigmoid'))

            model2.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

            # Merging models
            mergedOut = keras.layers.Add()([model.output,model2.output])
            mergedOut = Flatten()(mergedOut)
            mergedOut = Dense(64,activation='relu')(mergedOut)
            mergedOut = Dropout(.2)(mergedOut)
            mergedOut = Dense(64, activation='relu')(mergedOut)
            mergedOut = Dropout(.1)(mergedOut)

            # Output layer
            mergedOut = Dense(4,activation='softmax')(mergedOut)

            # Final merged model
            mergedModel = keras.models.Model([model.input,model2.input],mergedOut)

            mergedModel.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

            prev_time = time.time()

            mergedModel.fit([X,Y],Z,batch_size=64,epochs=20,validation_split=0.1, callbacks=[tensorboard])
            #model.fit(X,Y, batch_size=64,epochs=20, validation_split=0.1, callbacks=[tensorboard])
            print("Training took: {}".format(int(time.time() - prev_time)))
            mergedModel.save("model/64x3x1-CNN.model")

#model.save("model/{}x{}x{}-CNN.model".format(layer_size,conv_layer,dense_layer))