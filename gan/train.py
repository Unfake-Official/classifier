import os
from trainer import Trainer
from dcgan_discriminator import DCGAN_Discriminator
from dcgan_generator import DCGAN_Generator
from classifier import Classifier
from keras import layers, utils, Sequential

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

IMG_SIZE = (512, 256)

G_CHECKPOINT_PATH = 'gan/checkpoints/generator'
D_CHECKPOINT_PATH = 'gan/checkpoints/discriminator'
C_CHECKPOINT_PATH = 'gan/checkpoints/classifier'
METRICS_PATH = 'gan/metrics/metrics.png'

generator = DCGAN_Generator()
discriminator = DCGAN_Discriminator()
classifier = Classifier()

if os.path.exists(G_CHECKPOINT_PATH):
    generator = Sequential([layers.TFSMLayer(G_CHECKPOINT_PATH, call_endpoint='serving_default')])
    print('Generator loaded successfully')

if os.path.exists(D_CHECKPOINT_PATH):
    discriminator = Sequential([layers.TFSMLayer(D_CHECKPOINT_PATH, call_endpoint='serving_default')])
    print('Discriminator loaded successfully')

if os.path.exists(C_CHECKPOINT_PATH):
    classifier = Sequential([layers.TFSMLayer(C_CHECKPOINT_PATH, call_endpoint='serving_default')])
    print('Classifier loaded successfully')

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

# todo: Configure dataset for performance (cache and prefetch)
train_ds, test_ds = utils.image_dataset_from_directory(
    DATASET_PATH,
    label_mode='categorical',
    color_mode='rgb',
    validation_split=VALIDATION_SPLIT,
    subset='both',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

trainer = Trainer(classifier=classifier, generator=generator, discriminator=discriminator)
trainer.train(epochs=EPOCHS, train_ds=train_ds, test_ds=test_ds,
              generator_checkpoint_path=G_CHECKPOINT_PATH,
              discriminator_checkpoint_path=D_CHECKPOINT_PATH,
              classifier_checkpoint_path=C_CHECKPOINT_PATH,
              metrics_path=METRICS_PATH)
