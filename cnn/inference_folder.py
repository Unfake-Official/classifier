from classifier import Classifier
from keras import utils, models, metrics
from tqdm import tqdm

CHECKPOINT_PATH = 'cnn/checkpoints/model.keras'
FOLDER_PATH = r'folder_path'

model = models.load_model(CHECKPOINT_PATH)

ds = utils.image_dataset_from_directory(
    FOLDER_PATH,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    validation_split=0,
    seed=123,
    image_size=(512, 256))

loss = metrics.CategoricalCrossentropy(name='loss')
accuracy = metrics.CategoricalAccuracy(name='accuracy')

for images, labels in tqdm(ds):
    predictions = model(images, training=False)
    predictions = predictions['output_0']

    loss(labels, predictions)
    accuracy(labels, predictions)

print(loss.result())
print(accuracy.result())
