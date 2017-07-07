import utils
import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

import matplotlib.image as mpimg

# import h5py
# f = h5py.File('model.h5', 'r+')
# del f['optimizer_weights']
# f.close()

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        #steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image.save('image.jpg')

        #this is a hack - TODO: clean up
        image_path  = 'image.jpg'
        image = mpimg.imread(image_path)

        #resize
        image = utils.resize_image(image)

        x = []
        x.append(image)
        x = np.array(x)

        #predict
        predict = model.predict(x, batch_size=1, verbose=1)

        steering_angle = float(predict.item(0))
        #The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = 0.32

        if steering_angle > 0.04 and steering_angle <= 0.19:
            throttle = 0.25
        elif steering_angle >= 0.2:
            throttle = 0.1
        elif steering_angle < -0.04 and steering_angle >= -0.19:
            throttle = 0.25
        elif steering_angle <= -0.2:
            throttle = 0.1
        
        
        print(steering_angle, throttle)
        send_control(steering_angle*2, throttle)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    model = load_model('model2.h5')
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
