import cv2 
import d3dshot
import tensorflow as tf
import numpy as np
import time
import keyboard
from directkeys import PressKey, ReleaseKey, W, A, S, D

# Check if tensorflow-gpu is installed.
if tf.config.list_physical_devices('GPU'):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.2)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

cvNet = cv2.dnn.readNetFromTensorflow('opencv_model/frozen_inference_graph.pb', 'opencv_model/model.pbtxt')
to_break = False

pressedW = False
pressedA=False
pressedS=False
pressedD=False

def get_screen(d3d):
    global to_break
    image = np.array(d3d.screenshot())
    image = cv2.resize(image,(256,144))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def reshape(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    reshaped = image.reshape(-1, 144,256,1)
    return reshaped

def detect_objects(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    cvNet.setInput(cv2.dnn.blobFromImage(image, size=(256,144),swapRB=True,crop=False))
    cvOut = cvNet.forward()
    return cvOut

def get_objects(objects):
    # This is awful, later on don't make max detections but differently sized arrays
    # Adding bounding box to array of detected objects
    detected = []
    max_detections = 5
    actual_detection = 0
    for detection in objects[0,0,:,:]:
        score = float(detection[2])
        if score > 0.4:
            left = int(detection[3] * image.shape[1])
            top = int(detection[4] * image.shape[0])
            right = int(detection[5] * image.shape[1])
            bottom = int(detection[6] * image.shape[0])
            detected.append([left, top, right, bottom])
            actual_detection+=1
        if actual_detection == max_detections:
            return detected
            break

    # When 5 things aren't detected then it will fill the rest with zeroes
    while actual_detection < max_detections:
        detected.append([int(0),int(0),int(0),int(0)])
        actual_detection+=1
    detected = np.array(detected)
    return detected

def move(keys):
    if keys[0] == 0:
        ReleaseKey(W)
    else:
        PressKey(W)
    if keys[1] == 0:
        ReleaseKey(A)
    else:
        PressKey(A)
    if keys[2] == 0:
        ReleaseKey(S)
    else:
        PressKey(S)
    if keys[3] == 0:
        ReleaseKey(D)
    else:
        PressKey(D)

model = tf.keras.models.load_model("model/64x3x1-CNN.model")
d = d3dshot.create()

time.sleep(10)
print('starting')

pause = False
while to_break==False:
    if keyboard.is_pressed('o'):
        pause = True
        move([0,0,0,0])
    if keyboard.is_pressed('i'):
        pause = False

    if keyboard.is_pressed('q'):
        to_break = True
        move([0,0,0,0])
        break

    if pause == False:
        screen = get_screen(d)
        image = reshape(screen)

        # Getting objects
        objects = detect_objects(image)
        detected = get_objects(objects)

        prediction = model([image,detected[np.newaxis,:,:]])
        print("Forward-{} Left-{} Backward-{} Right-{}".format(prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3]))
        press = prediction[0]
        move(press)