import numpy as np
import mss
import cv2
import time
import keyboard

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
to_break = False

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

def get_key():
    presses = [keyboard.is_pressed('w'),keyboard.is_pressed('a'),keyboard.is_pressed('s'),keyboard.is_pressed('d')]
    return presses

time.sleep(0)
while to_break == False:
    screen = get_screen()
    keys = get_key()
    print(keys)