import os
import numpy as np
import mss
import cv2
import time
import keyboard

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
to_break = False

inputs = []
outputs = []

def get_screen():
    global to_break
    with mss.mss() as sct:
        printscreen = np.array(sct.grab(monitor))
        printscreen = cv2.resize(printscreen,(256,144))
        cv2.imshow('window', printscreen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            keyboard.unhook_all()
            to_break = True
    return printscreen

def get_key():
    presses = [keyboard.is_pressed('w'),keyboard.is_pressed('a'),keyboard.is_pressed('s'),keyboard.is_pressed('d')]
    return presses

def save_data(images, keys, number):
    i = 0
    while i < len(keys):
        path = "D:/projects/python scripts/ml/Forza_autopilot/collected_data/"
        cv2.imwrite("{}collected_data{}/images/image{}.jpg".format(path,number,i), images[i])

        out = keys[i]
        f = open("{}collected_data{}/keys/key{}.txt".format(path,number,i), "w+")
        f.write("{},{},{},{}".format(int(out[0]==True),int(out[1]==True),int(out[2]==True),int(out[3]==True)))
        f.close()
        i+=1

def create_paths(number):
    path = "D:/projects/python scripts/ml/Forza_autopilot/collected_data/"
    try:
        os.makedirs("{}collected_data{}/keys".format(path, number))
        os.makedirs("{}collected_data{}/images".format(path, number))
    except OSError:
        print("can't create directory")
    else:
        print("created directory")

number_of_session = input("number of actual session: ")
create_paths(int(number_of_session))
time.sleep(10)
while to_break == False:
    input_ = get_screen()
    output_ = get_key()
    print(output_)

    inputs.append(input_)
    outputs.append(output_)
save_data(inputs,outputs, int(number_of_session))

# labeling https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
# or this https://datascience.stackexchange.com/questions/49094/how-to-transform-a-folder-of-images-into-csv-file