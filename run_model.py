import cv2 
import mss
import tensorflow as tf
import numpy as np
import time
import keyboard
from directkeys import PressKey, ReleaseKey, W, A, S, D

# Check if tensorflow-gpu is installed.
if tf.test.gpu_device_name():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.2)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
to_break = False

pressedW = False
pressedA=False
pressedS=False
pressedD=False

def get_screen():
    global to_break
    with mss.mss() as sct:
        printscreen = np.array(sct.grab(monitor))
        printscreen = cv2.resize(printscreen,(256,144))
    return printscreen

def reshape(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    reshaped = image.reshape(-1, 144,256,1)
    return reshaped

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

time.sleep(2)
print('starting')

pause = False
while to_break==False:
    if keyboard.is_pressed('o'):
        pause = True
        move([0,0,0,0])
    if keyboard.is_pressed('i'):
        pause = False

    if pause == False:
        screen = get_screen()
        image = reshape(screen)

        prediction = model.predict([image])
        print("Forward-{} Left-{} Backward-{} Right-{}".format(prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3]))
        press = prediction[0]
        move(press)

        if keyboard.is_pressed('q'):
            to_break = True
            break