from classifier import Classifier
import numpy as np
from keras import utils, ops, Sequential, layers

CHECKPOINT_PATH = 'cnn/checkpoints/model'

model = Classifier()
model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])

IMG_PATH = r'C:\Users\mcsgo\Downloads\real.png'
IMG_SIZE = (512, 256)

class_names=['fake', 'real']

img = utils.load_img(IMG_PATH, grayscale=True, target_size=IMG_SIZE)
img_array = utils.img_to_array(img)

predictions = model.predict(img_array)
score = ops.nn.softmax(predictions[0])

print(f"Your audio is probably '{class_names[np.argmax(score)]}' with {100*np.max(score)}% confidence.")
