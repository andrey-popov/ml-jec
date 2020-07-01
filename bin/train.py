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

    lr = config['train'].get('learning_rate', None)
    if lr is not None:
        model.optimizer.lr.assign(lr)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model'),
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            write_graph=False, histogram_freq=1,
            profile_batch='500,520'
        )
    ]
    if 'reduce_lr_on_plateau' in config['train']:
        c = config['train']['reduce_lr_on_plateau']
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            factor=c['factor'], patience=c['patience'],
            min_delta=c['min_delta'], min_lr=c.get('min_lr', 0),
            verbose=1
        ))
    if 'early_stopping' in config['train']:
        c = config['train']['early_stopping']
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=c['patience'], min_delta=c['min_delta'],
            verbose=1
        ))

    history = model.fit(
        train_dataset, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
        validation_data=val_dataset, callbacks=callbacks
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
