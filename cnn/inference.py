from classifier import Classifier
import numpy as np
from keras import utils, ops, Sequential, layers
import tensorflow as tf
import librosa
from io import BytesIO
import matplotlib.pyplot as plt
import PIL.Image as img

CHECKPOINT_PATH = 'cnn/checkpoints/model'

IMG_SIZE = (256, 256)

model = Classifier()
model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])

AUDIO_PATH = r'C:\Users\mcsgo\OneDrive\Documentos\TCC\ASVSPOOF\LA\LA\ASVspoof2019_LA_eval\flac\LA_E_1000147.flac'

class_names = ['fake', 'real']

y, sr = librosa.load(AUDIO_PATH, sr=None)

spec = np.abs(librosa.cqt(y, sr=sr))
spec = librosa.amplitude_to_db(spec, ref=np.max)

img_bytesio = BytesIO()
plt.imsave(img_bytesio, spec, cmap='gray', format='png')
img_bytesio.seek(0)

image = img.open(img_bytesio)
image = image.resize(IMG_SIZE)

img_array = utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
img_array = tf.reshape(img_array, [-1, 256, 256, 1])

predictions = model.predict(img_array)
score = ops.nn.softmax(predictions['output_0'][0])

print(f"Your audio is probably '{class_names[np.argmax(score)]}' with {100*np.max(score):.2f}% confidence.")
