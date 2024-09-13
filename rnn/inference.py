from classifier import Classifier
import numpy as np
from keras import utils, models, ops

CHECKPOINT_PATH = 'rnn/checkpoints/model'

model = Classifier()
model = models.load_model(CHECKPOINT_PATH)

IMG_PATH = r'C:\Users\mcsgo\Downloads\real.png'
IMG_SIZE = (256, 256)

class_names=['fake', 'real']

img = utils.load_img(IMG_PATH, grayscale=True, target_size=IMG_SIZE)
img_array = utils.img_to_array(img)
img_array = ops.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = ops.nn.softmax(predictions[0])

print(f"Your audio is probably '{class_names[np.argmax(score)]}' with {100*np.max(score)}% confidence.")
