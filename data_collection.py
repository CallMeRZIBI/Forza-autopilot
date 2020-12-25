import numpy as np
from PIL import ImageGrab
import mss
import cv2
import time

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def screen_record():
    last_time = time.time()
    with mss.mss() as sct:
        while(True):
            printscreen = np.array(sct.grab(monitor))
            printscreen = cv2.resize(printscreen,(256,144))
            print("loop took {} seconds".format(time.time()-last_time))
            last_time = time.time()
            cv2.imshow('window', printscreen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

screen_record()