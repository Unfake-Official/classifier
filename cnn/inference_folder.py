from classifier import Classifier
from keras import utils, Sequential, layers, metrics
from tqdm import tqdm

CHECKPOINT_PATH = 'cnn/checkpoints/model'
FOLDER_PATH = 'C:\Users\mcsgo\OneDrive\Documentos\TCC\VCTK-Corpus-SPEC'

model = Classifier()
model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])

_, test_ds = utils.image_dataset_from_directory(
    FOLDER_PATH,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    validation_split=1,
    subset='both',
    seed=123,
    image_size=(512, 256))

IMG_PATH = r'C:\Users\mcsgo\Downloads\real.png'
IMG_SIZE = (512, 256)

loss = metrics.MeanSquaredError(name='test_loss')
accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

for images, labels in tqdm(test_ds):
    predictions = model(images, training=False)

    loss(labels, predictions)
    accuracy(labels, predictions)

print(loss.result())
print(accuracy.result())
