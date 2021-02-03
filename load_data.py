import numpy as np
import os
import cv2
import random
import tensorflow as tf
import gc
import time

training_data = []
cvNet = cv2.dnn.readNetFromTensorflow('opencv_model/frozen_inference_graph.pb', 'opencv_model/model.pbtxt')

def detect_objects(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    cvNet.setInput(cv2.dnn.blobFromImage(image, size=(256,144),swapRB=True,crop=False))
    cvOut = cvNet.forward()
    return cvOut

def get_objects(objects,image):
    # Adding bounding box to array of detected objects
    detected = []
    for detection in objects[0,0,:,:]:
        if detection[1] >=2 and detection[1] <= 9:
            score = float(detection[2])
            if score > 0.5:
                left = int(detection[3] * image.shape[1])
                top = int(detection[4] * image.shape[0])
                right = int(detection[5] * image.shape[1]) - left
                bottom = int(detection[6] * image.shape[0]) - top
                detected.append([left, top, right, bottom])

    return detected

def draw_boxes(coords):
    image = np.zeros((144,256,1), np.uint8)
    for detection in coords:
        cv2.rectangle(image,(detection[0],detection[1],detection[2],detection[3]), (255), thickness=10)
    image = cv2.resize(image,(64,36))
    return image

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
                image = cv2.imread(os.path.join(img_path,"image{}.jpg".format(i)))

                # Turning off garbage collector
                gc.disable()
                objects = detect_objects(image)

                detected = get_objects(objects, image)

                img = draw_boxes(detected)

                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                f = open(os.path.join(labels_path,"key{}.txt".format(i)), "r")
                f = f.read()
                label = f.split(',')
                label = [int(label[0]), int(label[1]),int(label[2]), int(label[3])]
                
                training_data.append([gray_image,label,img])
                gc.enable()
                
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

for image, label, objects in training_data:
    X.append(image)
    Y.append(label)
    Z.append(objects)

X = np.array(X).reshape(-1, 144, 256,1)
Y = np.array(Y)
Z = np.array(Z).reshape(-1,36,64,1)

# Saving data
np.savez_compressed("training_data/data.npz",X,Y,Z)
