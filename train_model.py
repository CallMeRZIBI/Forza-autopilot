'''import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard
import time

# Loading data
data = np.load("training_data/data.npz")
X = data['arr_0']

Y = data['arr_1']

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

            prev_time = time.time()

            model.fit(X,Y, batch_size=64,epochs=20, validation_split=0.1, callbacks=[tensorboard])
            print("Training took: {}".format(int(time.time() - prev_time)))

model.save("model/{}x{}x{}-CNN.model".format(layer_size,conv_layer,dense_layer))'''





#------------------neural network with two networks added together-----------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.callbacks import TensorBoard
from keras.layers.merge import concatenate
import time

# Loading data
data = np.load("training_data/data.npz",allow_pickle=True)
X = data['arr_0']

Y = data['arr_2']

Z = data['arr_1']

X = X/255.0

Y = Y/255.0

#--------------Optimization--------------
dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            Name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size,dense_layer, int(time.time()))
            print(Name)
            # TensorBoard
            tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

            # Model for image processing
            # First layer
            input1 = Input(shape=X.shape[1:])
            m = Conv2D(layer_size, (3,3))(input1)
            m = Activation("relu")(m)
            m = MaxPooling2D(pool_size=(2,2))(m)

            # Hidden layers
            for l in range(conv_layer-1):
                m = Conv2D(layer_size, (3,3))(m)
                m = Activation("relu")(m)
                m = MaxPooling2D(pool_size=(2,2))(m)
            flatten1 = Flatten()(m)

            '''for l in range(dense_layer):
                m = Dense(layer_size)(m)
                m = Activation('relu')(m)
                m = Dropout(0.2)(m)

            output1 = Dense(4,activation='sigmoid')(m)'''

#----------------------------FIX--------------------------------------

            # Model for object processing
            # Inputs
            input2 = Input(shape=Y.shape[1:])
            x = Conv2D(32,(3,3),input_shape=Y.shape[1:])(input2)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2,2))(x)

            for i in range(2):
                x = Conv2D(32,(3,3))(x)
                x = Activation("relu")(x)
                x = MaxPooling2D(pool_size=(2,2))(x)
            flatten2 = Flatten()(x)

            '''x = Dense(32)(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)

            output2 = Dense(4,activation='sigmoid')(x)'''

            # Merging models
            #concatenated = concatenate([output1,output2])
            concatenated = Concatenate()([flatten1, flatten2])
            l = Dense(64+32)(concatenated)
            l = Activation('relu')(l)
            l = Dropout(0.2)(l)
            output = Dense(4,activation='sigmoid')(l)

            merged_model = Model([input1, input2], output)
            merged_model.compile(loss='binary_crossentropy',
                    optimizer = 'adam',
                    metrics=['accuracy'])

            print(merged_model.summary())
            prev_time = time.time()

            merged_model.fit([X,Y],Z, batch_size=64,epochs=20,validation_split=0.5,callbacks=[tensorboard])
            #model.fit(X,Y, batch_size=64,epochs=20, validation_split=0.1, callbacks=[tensorboard])
            print("Training took: {}".format(int(time.time() - prev_time)))
            merged_model.save("model/{}x{}x{}-CNN.model".format(layer_size, conv_layer, dense_layer))

#model.save("model/{}x{}x{}-CNN.model".format(layer_size,conv_layer,dense_layer))
