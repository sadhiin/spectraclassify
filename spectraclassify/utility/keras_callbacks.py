import os
import time
import tensorflow as tf
from spectraclassify import logger, get_unique_file_name
from spectraclassify.utility.config_manager import get_model_conf

model_conifg = get_model_conf()

def get_log_path(DIR = "Tensorboard_logs/logs/fit"):
    log_file_name = time.strftime("TB_log_%Y_%m_%d-%H_%M_%S")
    os.makedirs(DIR, exist_ok=True)
    log_path = os.path.join(DIR, log_file_name)
    logger.info(f"Tensorboard log path: {log_path}")
    print(f"Tensorboard log path: {log_path}")
    return log_path

def get_callbacks():
    log_path = get_log_path()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=get_unique_file_name("Model", "h5"), save_best_only=True, monitor='val_loss', mode='min'),

        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),

        tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    ]

    return callbacks
