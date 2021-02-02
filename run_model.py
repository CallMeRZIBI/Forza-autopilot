'''import cv2 
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

def get_screen(d3d):
    global to_break
    image = np.array(d3d.screenshot())
    image = cv2.resize(image,(256,144))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def reshape(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2GRAY)
    #cv2.imshow('img',image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    reshaped = image.reshape(-1, 144,256,1)
    return reshaped

def detect_objects(image):
    img = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(256,144),swapRB=True,crop=False))
    cvOut = cvNet.forward()
    return cvOut

def get_objects(objects,image):
    # Adding bounding box to array of detected objects
    for detection in objects[0,0,:,:]:
        if detection[1] >=2 and detection[1] <= 9:
            score = float(detection[2])
            if score > 0.4:
                left = int(detection[3] * image.shape[1])
                top = int(detection[4] * image.shape[0])
                right = int(detection[5] * image.shape[1]) - left
                bottom = int(detection[6] * image.shape[0]) - top
                cv2.rectangle(image,(left,top,right,bottom), (0,0,0), thickness=3)

    return image

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
        
        detections = detect_objects(screen)
        image = get_objects(detections,screen)
        gray = reshape(image)

        prediction = model.predict([gray])
        print("Forward-{} Left-{} Backward-{} Right-{}".format(prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3]))
        press = prediction[0]
        move(press)'''





#----------------running model with multiple networks added together-----------------------------
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

def get_objects(objects, image):
    detected = []
    for detection in objects[0,0,:,:]:
        score = float(detection[2])
        if score > 0.4:
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
    image = image.reshape(-1,36,64,1)
    return image

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
        detected = get_objects(objects, image)
        boxes_im = draw_boxes(detected)

        prediction = model([image,boxes_im])
        print("Forward-{} Left-{} Backward-{} Right-{}".format(prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3]))
        press = prediction[0]
        move(press)