import os
from model import VisionTransformer
from trainer import Trainer
from keras import layers, utils, Sequential

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

IMG_SIZE = (256, 256)

CHECKPOINT_PATH = 'vision_transformer/checkpoints/model'
METRICS_PATH = 'vision_transformer/metrics/metrics.png'
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
DATASET_PATH = r'C:\Users\mcsgo\OneDrive\Documentos\TCC\Dataset'

model = VisionTransformer(IMG_SIZE[0], 4, 6, 3, 64, 4, 128, 1, 0.25)
if os.path.exists(CHECKPOINT_PATH):
    model = Sequential([layers.TFSMLayer(CHECKPOINT_PATH, call_endpoint='serving_default')])
    print('Model loaded successfully')

# todo: Configure dataset for performance (cache and prefetch)
train_ds, test_ds = utils.image_dataset_from_directory(
    DATASET_PATH,
    label_mode='categorical',
    color_mode='grayscale',
    validation_split=VALIDATION_SPLIT,
    subset='both',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

trainer = Trainer(model=model)
trainer.train(epochs=EPOCHS, train_ds=train_ds, test_ds=test_ds, checkpoint_path=CHECKPOINT_PATH, metrics_path=METRICS_PATH)
