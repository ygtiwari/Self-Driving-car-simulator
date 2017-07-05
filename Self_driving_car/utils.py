import numpy as np
import cv2 as cv
import csv
import random

def load_data(PATH):
    culled_list = []
    with open(PATH) as f:
        reader = csv.reader(f)
        initial_list = list(reader)

    #remove header
    initial_list.pop(0)

    #remove ones - they throw an error
    for index, item in enumerate(initial_list):
        #print(index, item[3])
        if(float(item[3])>=1):
            initial_list.pop(index)

    culled_list = initial_list;

    return culled_list

def resize_image(image):
    height = int(image.shape[0]/1)
    width = int(image.shape[1]/1)
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def flip_image(image):
    return cv.flip(image,1)

def flip_angle(angle):
    return angle*-1

def bright_image(image):
    bright_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    bright_image[:, :, 2] = bright_image[:, :, 2] * random_bright
    image1 = cv.cvtColor(bright_image, cv.COLOR_HSV2RGB)
    return image1


