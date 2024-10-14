import numpy as np
from keras import utils, ops, models
import tensorflow as tf
import librosa
from io import BytesIO
import matplotlib.pyplot as plt
import PIL.Image as img

CHECKPOINT_PATH = 'cnn/checkpoints/model.keras'

IMG_SIZE = (512, 256)

model = models.load_model(CHECKPOINT_PATH)

AUDIO_PATH = r'audio_path'

class_names = ['fake', 'real']

y, sr = librosa.load(AUDIO_PATH, sr=None)

spec = np.abs(librosa.cqt(y, sr=sr))
spec = librosa.amplitude_to_db(spec, ref=np.max)

img_bytesio = BytesIO()
plt.imsave(img_bytesio, spec, cmap='viridis')
img_bytesio.seek(0)

image = img.open(img_bytesio).convert('RGB')
image = image.resize(IMG_SIZE)

img_array = utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

print(f"Your audio is probably '{class_names[np.argmax(predictions)]}'")
