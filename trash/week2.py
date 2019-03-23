import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import *
import PIL.Image as img


data = np.array(img.open('2222.png').convert('L').resize((5, 5), img.ANTIALIAS))
data = data / 255.0
data1 = np.array([np.array(img.open('2222.png').convert('L').resize((5, 5), img.ANTIALIAS))])
data1 = data1 / 255.0
print(data)
print(data1)
