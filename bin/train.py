#!/usr/bin/env python

"""Trains NN according to the given configuration."""

import argparse
import math
import os
import pickle
from typing import Dict, List, Mapping, Sequence

import numpy as np
import tensorflow as tf
import yaml

from mljec import build_datasets, build_model, plot_history, summarize_model


def create_piecewise_linear_schedule(
    epochs: Sequence[int], learning_rate: Sequence[float]
) -> tf.keras.callbacks.LearningRateScheduler:
    """Create piecewise linear learning rate scheduler.

    Args:
        epochs:  Ordered sequence of zero-based epoch indices.
        learning_rate:  Values of learning rate at each epoch specified
            in the first argument.

    Return:
        Scheduler callback.
    """

    if any(epochs[i] >= epochs[i + 1] for i in range(len(epochs) - 1)):
        raise RuntimeError('Sequence of epochs is not ordered.')
    if len(epochs) != len(learning_rate):
        raise RuntimeError(
            'Mismatched numbers of epochs and values for learning rate.'
        )
    if len(epochs) < 2:
        raise RuntimeError('Must provide at least two points.')

    def schedule(epoch, _):
        i = 0
        while i < len(epochs) - 1 and epoch > epochs[i + 1]:
            i += 1
        frac = (epoch - epochs[i]) / (epochs[i + 1] - epochs[i])
        return (
            (learning_rate[i + 1] - learning_rate[i]) * frac
            + learning_rate[i]
        )

    return tf.keras.callbacks.LearningRateScheduler(schedule)


def train(
    config: Mapping, model: tf.keras.Model,
    train_dataset: tf.data.Dataset, train_dataset_size: int,
    val_dataset: tf.data.Dataset, output_dir: str = ''
) -> Dict[str, List[float]]:
    """Train a model.

    Args:
        config:  Master configuration.
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
    if 'piecewise_linear_schedule' in config['train']:
        for key in [
            'max_epochs', 'learning_rate', 'reduce_lr_on_plateau',
            'early_stopping'
        ]:
            if key in config['train']:
                raise RuntimeError(
                    f'Cannot use key "{key}" together with '
                    '"piecewise_linear_schedule".'
                )
        c = config['train']['piecewise_linear_schedule']
        callbacks.append(create_piecewise_linear_schedule(
            c['epochs'], c['learning_rate']
        ))
        max_epochs=c['epochs'][-1] + 1

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
    if 'cache' in config['data']:
        cache = config['data']['cache']
        if cache is not False and cache is not None:
            train_dataset = train_dataset.cache(cache)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.shuffle(100)
    val_dataset = val_dataset.cache()

    model = build_model(config, metadata['cardinalities'])
    summarize_model(
        model, fig_dir=args.output if args.plot_model else None
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
