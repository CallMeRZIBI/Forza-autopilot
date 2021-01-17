import numpy as np
import os
import cv2
import random

training_data = []
cvNet = cv2.dnn.readNetFromTensorflow('opencv_model/frozen_inference_graph.pb', 'opencv_model/model.pbtxt')

def detect_objects(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    cvNet.setInput(cv2.dnn.blobFromImage(image, size=(256,144),swapRB=True,crop=False))
    cvOut = cvNet.forward()
    return cvOut

def create_training_data():
    try:
        actual_file = 0
        for files in os.listdir("collected_data"):
            path = os.path.join("collected_data", files)
            img_path = os.path.join(path, "images")
            labels_path = os.path.join(path, "keys")

            actual_file += 1
            i = 0
            while i < len(os.listdir(img_path)):
                detected = []
                image = cv2.imread(os.path.join(img_path,"image{}.jpg".format(i)))
                objects = detect_objects(image)
                for detection in objects[0,0,:,:]:
                    score = float(detection[2])
                    if score > 0.4:
                        left = detection[3] * image.shape[1]
                        top = detection[4] * image.shape[0]
                        width = (detection [5] * image.shape[1]) - left
                        height = (detection[6] * image.shape[0]) - top
                        detected.append([left, top, width, height])

                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                f = open(os.path.join(labels_path,"key{}.txt".format(i)), "r")
                f = f.read()
                label = f.split(',')
                label = [int(label[0]), int(label[1]),int(label[2]), int(label[3])]
                training_data.append([gray_image,label,detected])
                i+=1
                
                print("Loaded: {} out of {}, {} folder".format(i,len(os.listdir(img_path)),actual_file,len(files)))
    except Exception as e:
        print(e)

create_training_data()
# Maybye shuffle training data
random.shuffle(training_data)

#-----------------data loaded----------------

X = []
Y = []
Z = []

for image, label, detected in training_data:
    X.append(image)
    Y.append(label)
    Z.append(detected)

X = np.array(X).reshape(-1, 144, 256,1)
Y = np.array(Y)
Z = np.array(Z, dtype=object)

# Saving data
np.savez_compressed("training_data/data.npz",X,Y,Z)