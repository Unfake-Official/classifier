from model import VisionTransformer
import numpy as np
from keras import layers, utils, ops, Sequential

CHECKPOINT_PATH = 'cnn/checkpoints/model'

model = VisionTransformer()
model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])

IMG_PATH = r'img_path'
IMG_SIZE = (512, 256)

class_names=['fake', 'real']

img = utils.load_img(IMG_PATH, grayscale=True, target_size=IMG_SIZE)
img_array = utils.img_to_array(img)
img_array = ops.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = ops.nn.softmax(predictions['output_0'])

print(f"Your audio is probably '{class_names[np.argmax(score)]}' with {100*np.max(score)}% confidence.")
