from collections.abc import Sequence
import math

import tensorflow as tf
import tensorflow.keras.layers as layers


def build_model(config, num_features):
    """Construct model specified in the configuration.

    Also create optimizer and set the loss function.

    Args:
        config:  Dictionary representing configuration file.
        num_features:  Number of input features for the model.

    Return:
        Compiled model.
    """

    model_config = config['model']
    model_type = model_config['type']
    if model_type == 'plain':
        model = _build_model_plain(
            num_layers=model_config.get('num_layers', None),
            num_units=model_config['num_units'],
            batch_norm=model_config.get('batch_norm', False),
            dropout=model_config.get('dropout', 0),
            num_features=num_features
        )
    else:
        raise RuntimeError(f'Unknown model type "{model_type}".')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['optimizer'].get('learning_rate', 1e-3)
    )
    if config['loss'] == 'huber':
        loss = tf.keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile(optimizer, loss=loss)
    return model


def _build_model_plain(
    num_layers=5, num_units=256, batch_norm=False, dropout=0,
    num_features=None, name=None
):
    """Construct an MLP.

    Args:
        num_layers:  Number of hidden layers.  Ignored if num_units is
            a sequence.
        num_units:  Number of units in each hidden layer.  Can be a
            sequence.  In that case it provides numbers of units in all
            hidden layers.
        batch_norm:  Whether to apply batch normalization.
        dropout:  Dropout rate.  Disabled if 0.
        num_features:  Optional number of input features.
        name:  Optional name for the model.
    """

    if not isinstance(num_units, Sequence):
        num_units = [num_units] * num_layers

    model = tf.keras.Sequential(name=name)
    for i, n in enumerate(num_units):
        model.add(layers.Dense(
            n, kernel_initializer='he_uniform', use_bias=not batch_norm,
            input_shape=(num_features,) if i == 0 else (None,)
        ))
        if batch_norm:
            model.add(layers.BatchNormalization(scale=False))
        model.add(layers.Activation('relu'))
        if dropout > 0 and i != 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    return model

