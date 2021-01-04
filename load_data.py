import numpy as np
import os
import cv2
import random

training_data = []

def create_training_data():
    try:
        for files in os.listdir("collected_data"):
            path = os.path.join("collected_data", files)
            img_path = os.path.join(path, "images")
            labels_path = os.path.join(path, "keys")
            i = 0
            while i < len(os.listdir(img_path)):
                image = cv2.imread(os.path.join(img_path,"image{}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)
                f = open(os.path.join(labels_path,"key{}.txt".format(i)), "r")
                f = f.read()
                label = f.split(',')
                label = [int(label[0]), int(label[1]),int(label[2]), int(label[3])]
                training_data.append([image,label])
                i+=1
    except Exception as e:
        print(e)

create_training_data()
# Maybye shuffle training data
random.shuffle(training_data)

#-----------------data loaded----------------

X = []
Y = []

for image, label in training_data:
    X.append(image)
    Y.append(label)
X = np.array(X).reshape(-1, 144, 256,1)
Y = np.array(Y)

# Saving data
np.savez_compressed("training_data/data.npz",X,Y)