#!/usr/bin/env python

"""Trains NN according to the given configuration."""

import argparse
import math
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from data import build_datasets
from model import build_model


def train(
    config, model,
    train_dataset, train_dataset_size, val_dataset,
    output_dir=''
):
    """Train a model.

    Args:
        config:  Directory representing configuration file.
        model:  Model to train.
        train_dataset:  Training dataset.
        train_dataset_size:  Number of examples in the training dataset.
        val_dataset:  Validation dataset.
        output_dir:  Directory in whose subdirectories to store model
            checkpoints and TensorBoard logs.

    Return:
        Dictionary with training history.
    """

    steps_per_epoch = config['train']['steps_per_epoch']
    max_epochs = config['train'].get('max_epochs', None)
    if max_epochs is None:
        # At maximum, will iterate 10 times over the training set
        batch_size = config['data']['batch_size']
        max_epochs = math.ceil(
            10 * train_dataset_size / (steps_per_epoch * batch_size))

    min_delta = config['train']['min_delta']
    history = model.fit(
        train_dataset, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, patience=10, min_delta=min_delta, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=15, min_delta=min_delta, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, 'model'),
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(output_dir, 'logs'),
                write_graph=False, histogram_freq=1, profile_batch=10
            )
        ]
    )
    return history.history


def plot_history(config, history, save_path, ylim=(None, None), zoom=False):
    """Plot training history.

    Plot the loss on the training and validation sets and mark points
    when the learning rate was adjusted.

    Args:
        config:  Directory representing configuration file.
        history:  Dictionary with training history.
        save_path:  Save path for produced figure.
        ylim:  Manual range for y axis.  If not given, an automatic
            range will be used.
        zoom:  If True, will skip the initial segment of training when
            computing the range for the y axis.  Cannot be used together
            with ylim.
    """

    freq = config['train']['steps_per_epoch']
    if ylim[0] is None and ylim[1] is None and zoom:
        # Find the range of values in the last 75% of epochs
        skip = round(0.25 * len(history['loss']))
        maximum = max(
            max(history['loss'][skip:]), max(history['val_loss'][skip:])
        )
        minimum = min(
            min(history['loss'][skip:]), min(history['val_loss'][skip:])
        )
        # Include a 5% margin on each side
        margin = 0.05 / (1 - 2 * 0.05) * (maximum - minimum)
        ylim = (minimum - margin, maximum + margin)

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_alpha(0)
    axes = fig.add_subplot(111)
    x = np.arange(1, len(history['loss']) + 1) * freq
    axes.plot(x, history['loss'], label='Training')
    axes.plot(x, history['val_loss'], label='Validation')
    lr_changes = [
        (i + 0.5) * freq for i in range(len(history['lr']) - 1)
        if history['lr'][i] != history['lr'][i + 1]
    ]
    for x in lr_changes:
        axes.axvline(x, ls='dashed', lw=0.8, c='gray')
    axes.set_ylim(ylim)
    axes.set_xlabel('Parameter updates')
    axes.set_ylabel('Loss')
    axes.legend()

    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('config', help='YAML configuration file.')
    arg_parser.add_argument(
        '-o', '--output', default='', help='Directory to store output.'
    )
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.output)
    except FileExistsError:
        pass

    print('GPU devices:', tf.config.list_physical_devices('GPU'))
    with open(args.config) as f:
        config = yaml.safe_load(f)

    metadata, train_dataset, val_dataset, test_dataset = build_datasets(config)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache()

    model = build_model(config, len(metadata['features']))
    print(model.summary())

    history = train(
        config, model,
        train_dataset, metadata['counts']['train'], val_dataset,
        args.output
    )
    print('Validation loss:', min(history['val_loss']))

    for zoom, postfix in [(False, ''), (True, '_zoomed')]:
        plot_history(
            config, history,
            os.path.join(args.output, f'history{postfix}.pdf'),
            zoom=zoom
        )
    with open(os.path.join(args.output, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    if config['predict']:
        predictions = model.predict(test_dataset)
        np.save(os.path.join(args.output, 'prediction.npy'), predictions[:, 0])

