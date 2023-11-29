#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
# homework-08 uses Conv2D model, try to extend keras_image_helper ?
# from keras_image_helper import create_preprocessor
from PIL import Image


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# replicate ImageDataGenerator(rescale=1./255)
def preprocess_input(x):
    return x / 255.0



target_size = (150, 150)
classes = [
    'bee',
    'wasp'
]
# url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


