from classifier import Classifier
from keras import utils, Sequential, layers, metrics
from tqdm import tqdm

CHECKPOINT_PATH = 'cnn/checkpoints/model'
FOLDER_PATH = r'folder_path'

model = Classifier()
model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])

ds = utils.image_dataset_from_directory(
    FOLDER_PATH,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    validation_split=0,
    seed=123,
    image_size=(512, 256))

loss = metrics.MeanSquaredError(name='loss')
accuracy = metrics.CategoricalAccuracy(name='accuracy')

for images, labels in tqdm(ds):
    predictions = model(images, training=False)
    predictions = predictions['output_0']

    loss(labels, predictions)
    accuracy(labels, predictions)

print(loss.result())
print(accuracy.result())
