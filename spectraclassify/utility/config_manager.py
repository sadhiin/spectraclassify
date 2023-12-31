# this file is responsive for the management of configuration of the model and the data is passed by the user
import os
import pathlib
import sys
import yaml
import json
import warnings
from spectraclassify import logger
from spectraclassify.utility.utils import load_json

training_dir = None
validation_dir = None
classes = None
model_name = None
img_size = None
h = None
w = None
channels = None
model_name = None
epochs = None
batch_size = None
augmentation = None
freeze_layer = None
learning_rate = None
optimizer = None
loss = None
__file__ = 'config.json'
try:
    __path = pathlib.Path(__file__).parent.absolute()
    PARAMS = load_json(__path)

    # user passed configuration data
    training_dir = PARAMS['Training_Dir']
    validation_dir = PARAMS['Validation_Dir']
    classes = PARAMS['Classes']
    model_name = PARAMS['Model_Name']
    img_size = PARAMS['Image_Size'].split(',')
    h = int(img_size[0])
    w = int(img_size[1])
    channels = int(img_size[2])
    model_name = PARAMS['Model_Name']
    epochs = PARAMS['Epochs']
    batch_size = PARAMS['Batch_Size']
    learning_rate = PARAMS['Learning_Rate']
    augmentation = PARAMS['Augmentation']
    freeze_layer = PARAMS['Freeze_Layer']
    optimizer = PARAMS['Optimizer']
    loss = PARAMS['Loss']

except Exception as err:
    logger.error(f" Error loading config.json: {err}")


def get_Data_conf(
        training_dir=training_dir,
        validation_dir=validation_dir,
        classes=classes,
        h=h, w=w, channels=channels):
    CONFIG = {
        'TRAINING_DIR': training_dir,
        'VALIDATION_DIR': validation_dir,
        'CLASSES': classes,
        'IMG_SIZE': (h, w, channels),
        'BATCH_SIZE': batch_size
    }
    return CONFIG


def get_model_conf(model_name=model_name, epochs=epochs, learning_rate=learning_rate, optimizer=optimizer, loss=loss):
    CONFIG = {
        'MODEL_NAME': model_name,
        'EPOCHS': epochs,
        'LEARNING_RATE': learning_rate,
        'OPTIMIZER': optimizer,
        'LOSS': loss
    }
    return CONFIG
