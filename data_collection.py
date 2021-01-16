import os
import numpy as np
import d3dshot
import cv2
import time
import keyboard
import psutil

p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
to_break = False

inputs = []
outputs = []

def get_screen(d3d):
    global to_break
    image = np.array(d3d.screenshot())
    image = cv2.resize(image,(256,144))
    cv2.imshow('window', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        keyboard.unhook_all()
        to_break = True
    return image

def get_key():
    presses = [keyboard.is_pressed('w'),keyboard.is_pressed('a'),keyboard.is_pressed('s'),keyboard.is_pressed('d')]
    return presses

def detect_objects(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    rows = image.shape[0]
    cols = image.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(image, size=(256,144),swapRB=True,crop=False))
    cvOut = cvNet.forward()
    return cvOut

def save_data(images, keys, number):
    i = 0
    while i < len(keys):
        cv2.imwrite("collected_data/collected_data{}/images/image{}.jpg".format(number,i), images[i])

        out = keys[i]
        f = open("collected_data/collected_data{}/keys/key{}.txt".format(number,i), "w+")
        f.write("{},{},{},{}".format(int(out[0]==True),int(out[1]==True),int(out[2]==True),int(out[3]==True)))
        f.close()
        i+=1

def create_paths(number):
    try:
        os.makedirs("collected_data/collected_data{}/keys".format(number))
        os.makedirs("collected_data/collected_data{}/images".format(number))
    except OSError:
        print("can't create directory")
    else:
        print("created directory")

number_of_session = input("number of actual session: ")
create_paths(int(number_of_session))

cvNet = cv2.dnn.readNetFromTensorflow('opencv_model/frozen_inference_graph.pb', 'opencv_model/model.pbtxt')
d = d3dshot.create()
time.sleep(10)

while to_break == False:
    input_ = get_screen(d)
    output_ = get_key()
    print(output_)
    objects = detect_objects(input_)

    inputs.append(input_)
    outputs.append(output_)

save_data(inputs,outputs, int(number_of_session))