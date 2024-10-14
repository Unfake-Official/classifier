import os
from keras import utils, models
from classifier import Classifier
from trainer import Trainer

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.01

IMG_SIZE = (512, 256)

CHECKPOINT_PATH = 'cnn/checkpoints/model.keras'
METRICS_PATH = 'cnn/metrics/metrics.png'
CSV_PATH = 'cnn/metrics/metrics.csv'
'''
dataset folder with the following structure:
main_directory/
    fake/
        img1
        img2
        ...
    real/
        img1
        img2
        ...
'''
DATASET_PATH = r'ds_path'

model = Classifier()
if os.path.exists(CHECKPOINT_PATH):
    model = models.load_model(CHECKPOINT_PATH)

# todo: Configure dataset for performance (cache and prefetch)
train_ds, test_ds = utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    validation_split=VALIDATION_SPLIT,
    subset='both',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

trainer = Trainer(model=model, learning_rate=LEARNING_RATE)
trainer.train(epochs=EPOCHS, train_ds=train_ds, test_ds=test_ds,
              checkpoint_path=CHECKPOINT_PATH, metrics_path=METRICS_PATH, csv_path=CSV_PATH)
