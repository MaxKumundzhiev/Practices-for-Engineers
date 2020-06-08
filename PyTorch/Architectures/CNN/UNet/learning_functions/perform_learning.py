# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

from learning_functions.create_partition import create_partition_and_labels
from learning_functions.data_generator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os
import inspect
import gc


def perform_learning(training_sample_dir, val_sample_dir,
                     batch_size, three_d, sample_channels, categorise, output_classes, shuffle,
                     model_func, model_params, epochs, model_path, checkpoint_path,
                     log_name):

    partition, labels = create_partition_and_labels(training_sample_dir, val_sample_dir)

    params = {'batch_size': batch_size,
              'three_d': three_d,
              'n_channels': sample_channels,
              'categorise': categorise,
              'n_classes': output_classes,
              'shuffle': shuffle}

    training_generator = DataGenerator(partition['train'], labels, training_sample_dir, **params)
    validation_generator = DataGenerator(partition['validation'], labels, val_sample_dir, **params)

    cropped_path = checkpoint_path[:checkpoint_path.rfind('/')]
    if not os.path.exists(cropped_path):
        os.makedirs(cropped_path)
    checkpoint = ModelCheckpoint(checkpoint_path, period=3)

    model = model_func(**model_params)

    now = datetime.datetime.now()
    tensorboard_name = now.strftime("%Y-%m-%d-%H:%M")
    tensorboard_name = log_name + '-' + tensorboard_name
    path = "logs/" + tensorboard_name
    tensorboard = TensorBoard(log_dir=path)
    if not os.path.exists(path):
        os.makedirs(path)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=-1,
                        epochs=epochs,
                        callbacks=[checkpoint, tensorboard])

    model.save(model_path)

    del model
    gc.collect()

