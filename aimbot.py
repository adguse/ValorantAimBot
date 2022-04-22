#import torch
#import os
#from yolov5 import train, val, detect, export
from yolov5.models.common import *
import cv2 as cv
import numpy as np
from time import time

from yolov5.helpers import YOLOv5
from windowcapture import WindowCapture

# set model params
#model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
#device = "cuda" # or "cpu"



# init yolov5 model
#yolov5 = YOLOv5(model_path, device)

mod = YOLOv5('C://Users//zumbr//Desktop//aimbot//yolov5//runs//train//exp11//weights//best.pt')


# initialize the WindowCapture class
#WindowCapture.list_window_names()
wincap = WindowCapture()

wincap.start()

loop_time = time()

while(True):

    # if we don't have a screenshot yet, don't run the code below this point yet
    if wincap.screenshot is None:
        continue

    halfimg = cv.resize(wincap.screenshot,(0, 0), fx=0.75, fy=0.75)

    result: Detections = mod.predict(halfimg)

    result.render()
    temp = result.imgs[0]
    cv.imshow('Computer Vision', temp)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
       # wincap.stop()
        cv.destroyAllWindows()
        break

print('Done.')
