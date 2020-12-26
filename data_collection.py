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

def save_data(images, keys):
    i = 0
    while i < len(keys):
        cv2.imwrite("D:/projects/python scripts/ml/Forza_autopilot/collected_data/images/image{}.jpg".format(i), images[i])

        out = keys[i]
        f = open("D:/projects/python scripts/ml/Forza_autopilot/collected_data/keys/key{}.txt".format(i), "w+")
        f.write("{},{},{},{}".format(int(out[0]==True),int(out[1]==True),int(out[2]==True),int(out[3]==True)))
        f.close()
        i+=1

time.sleep(10)
while to_break == False:
    input_ = get_screen()
    output_ = get_key()
    print(output_)

    inputs.append(input_)
    outputs.append(output_)
save_data(inputs,outputs)

# labeling https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
# or this https://datascience.stackexchange.com/questions/49094/how-to-transform-a-folder-of-images-into-csv-file