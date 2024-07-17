from model import VisionTransformer
import numpy as np
from keras import layers, utils
import tensorflow as tf

CHECKPOINT_PATH = 'cnn/checkpoints/model'

model = VisionTransformer()
model = layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')

IMG_PATH = r'C:\Users\mcsgo\Downloads\real.png'
IMG_SIZE = (256, 256)

class_names=['fake', 'other', 'real']

img = utils.load_img(IMG_PATH, grayscale=True, target_size=IMG_SIZE)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"Your audio is probably '{class_names[np.argmax(score)]}' with {100*np.max(score)}% confidence.")
