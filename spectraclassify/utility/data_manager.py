"""
This file is responsive for the management of the data passed by the user.
Such as:
    - Training and Validation data
    - Training and Validation with/without data augmentation
    - Input classes
    - Preprocess input image for prediction
"""

import numpy as np
from spectraclassify import logger
from spectraclassify.utiliy.config_manager import get_Data_conf
from tenforflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

userconfig = get_Data_conf()


def preprocess_input_image(x):
    try:
        x = image.load_img(x, target_size=userconfig['Image_Size'][:-1])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        return x
    except Exception as e:
        logger.error(f"Error in preprocess_input_image: {e}")
        print(f"Error in preprocess_input_image: {e}")
        return None


def data_generator(do_augmentation: bool = userconfig['Augmentation']):
    if do_augmentation:
        training_data_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)
    else:
        training_data_generator = ImageDataGenerator(rescale=1. / 255)
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)

    return training_data_generator, validation_data_generator


def get_data_generator(
        training_dir=userconfig['Training_Dir'],
        validation_dir=userconfig['Validation_Dir'],
        batch_size=userconfig['Batch_Size'],
        do_augmentation=userconfig['Augmentation']):

    try:
        training_data_generator, validation_data_generator = data_generator(
            do_augmentation)

        training_generator = training_data_generator.flow_from_directory(
            directory=training_dir,
            target_size=userconfig['Image_Size'][:-1],
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = validation_data_generator.flow_from_directory(
            directory=validation_dir,
            target_size=(userconfig['Image_Size'][0],
                         userconfig['Image_Size'][1]),
            batch_size=batch_size,
            class_mode='categorical')

        return training_generator, validation_generator

    except Exception as e:
        logger.error(f"Error in get_data_generator: {e}")
        print(f"Error in get_data_generator: {e}")
        return None, None


def input_classes():
    train, val = get_data_generator()
    if train is None:
        raise ValueError("Unable to load training data")

    print(f"input_classes: {train.class_indices}")
    return train.class_indices
