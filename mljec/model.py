from collections.abc import Sequence
import math
from typing import List, Mapping

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Dense, Dropout
)


def build_model(config: Mapping) -> keras.Model:
    """Construct model specified in the configuration.

    Also create optimizer and set the loss function.

    Args:
        config:  Dictionary representing configuration file.

    Return:
        Compiled model.
    """

    model_config = config['model']
    if isinstance(model_config, str):
        model = keras.models.load_model(model_config)
        return model

    features = config['features']
    num_features = len(features['global']['numeric'])

    head_config = model_config['head']
    inputs = keras.Input(shape=(num_features, ), name='global_numeric')
    model_type = head_config['type']
    assert head_config['num_units'][-1] == 1
    if model_type == 'mlp':
        out = _apply_mlp(
            inputs,
            num_units=head_config['num_units'][:-1],
            batch_norm=head_config.get('batch_norm', False),
            dropout=head_config.get('dropout', 0),
            name_prefix='head_'
        )
    elif model_type == 'resnet':
        out = _apply_resnet(
            inputs,
            num_units=head_config['num_units'][:-1],
            name_prefix='head_'
        )
    else:
        raise RuntimeError(f'Unknown model type "{model_type}".')
    out = Dense(1)(out)
    model = keras.Model(inputs=inputs, outputs=out)

    if config['loss'] == 'huber':
        loss = keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile('adam', loss=loss)
    return model


def _apply_mlp(
    inputs: tf.Tensor, num_units: List[int], batch_norm: bool = False,
    dropout: float = 0., name_prefix: str = ''
) -> tf.Tensor:
    """Apply an MLP fragment to given inputs.

    Args:
        inputs:  Inputs to the network fragment.
        num_units:  Numbers of units in each layer.
        batch_norm:  Whether to apply batch normalization.
        dropout:  Dropout rate.  Disabled if 0.
        name_prefix:  Name prefix for layers.
    """

    x = inputs
    for layer_index, layer_num_units in enumerate(num_units):
        x = Dense(
            layer_num_units, kernel_initializer='he_uniform',
            use_bias=not batch_norm,
            name=name_prefix + f'dense_{layer_index + 1}'
        )(x)
        if batch_norm:
            x = BatchNormalization(
                scale=False, name=name_prefix + f'batch_norm_{layer_index + 1}'
            )(x)
        x = Activation(
            'relu', name=name_prefix + f'activation_{layer_index + 1}'
        )(x)
        if dropout > 0.:
            x = Dropout(
                dropout, name=name_prefix + f'dropout_{layer_index + 1}'
            )(x)
    return x


def _apply_resnet(
    inputs: tf.Tensor, num_units: List[int],
    name_prefix: str = ''
) -> tf.Tensor:
    """Apply a ResNet fragment to given inputs.

    Args:
        inputs:  Inputs for the network fragment.
        num_units:  Numbers of units in each layer.
        name_prefix:  Name prefix for layers.

    The skip connections will include trainable projections if the
    dimensions of layers with odd indices don't match.
    """

    x = inputs

    for i in range(0, len(num_units) - 1, 2):
        n1 = num_units[i]
        n2 = num_units[i + 1]
        ilayer = i // 2 + 1

        x_main = Dense(
            n1, kernel_initializer='he_uniform',
            name=name_prefix + f'dense_{ilayer}_1'
        )(x)
        x_main = Activation(
            'relu', name=name_prefix + f'activation_{ilayer}_1'
        )(x_main)
        x_main = Dense(
            n2, kernel_initializer='he_uniform',
            name=name_prefix + f'dense_{ilayer}_2'
        )(x_main)

        if x.shape[-1] == n2:
            x_bypass = x
        else:
            # Include a projection to match the dimensions
            x_bypass = Dense(
                n2, kernel_initializer='he_uniform', use_bias=False,
                name=name_prefix + f'projection_{ilayer}'
            )(x)

        x = Add(name=name_prefix + f'add_{ilayer}')([x_main, x_bypass])
        x = Activation('relu', name=name_prefix + f'activation_{ilayer}_2')(x)

    if len(num_units) % 2 == 1:
        x = Dense(
            num_units[-1], kernel_initializer='he_uniform',
            name=name_prefix + 'dense_last'
        )(x)
        x = Activation('relu', name=name_prefix + 'activation_last')(x)

    return x
