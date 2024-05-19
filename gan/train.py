import os
import tensorflow as tf
from cnn.classifier import Classifier
from trainer import Trainer

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

IMG_SIZE = (256, 256)

G_CHECKPOINT_PATH = 'gan/checkpoints/generator'
D_CHECKPOINT_PATH = 'gan/checkpoints/discriminator'
C_CHECKPOINT_PATH = 'gan/checkpoints/classifier'
METRICS_PATH = 'cnn/metrics/metrics.png'
'''
dataset folder with the following structure:
main_directory/
    fake/
        img1
        img2
        ...
    other/
        img1
        img2
        ...
    real/
        img1
        img2
        ...
'''
DATASET_PATH = r'C:\Users\mcsgo\OneDrive\Documentos\Dataset'

# todo: Configure dataset for performance (cache and prefetch)
train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    label_mode='categorical',
    color_mode='grayscale',
    validation_split=VALIDATION_SPLIT,
    subset='both',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

trainer = Trainer()
trainer.train(epochs=EPOCHS, train_ds=train_ds, test_ds=test_ds,
              generator_checkpoint_path=G_CHECKPOINT_PATH,
              discriminator_checkpoint_path=D_CHECKPOINT_PATH,
              classifier_checkpoint_path=C_CHECKPOINT_PATH,
              metrics_path=METRICS_PATH)
