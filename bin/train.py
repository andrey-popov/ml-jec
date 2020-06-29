#!/usr/bin/env python

"""Trains NN according to the given configuration."""

import argparse
import math
import os
import pickle

import numpy as np
import tensorflow as tf
import yaml

from mljec import build_datasets, build_model, plot_history


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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('config', help='YAML configuration file.')
    arg_parser.add_argument(
        '-o', '--output', default='', help='Directory to store output.'
    )
    arg_parser.add_argument(
        '--plot-model', action='store_true',
        help='Request a plot of the model. Requires graphviz and pydot.'
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
    if args.plot_model:
        tf.keras.utils.plot_model(
            model, os.path.join(args.output, 'model.png'), dpi=300
        )

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
