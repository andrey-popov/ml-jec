from collections.abc import Sequence
import math
from typing import List, Mapping

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Dense, Dropout
)


class MaskedSum(keras.layers.Layer):
    """Layer that sums over temporal slices, with a mask.

    Input tensor is summed along axis 1, while only including indices
    enabled in the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        conv_mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        result = tf.math.reduce_sum(
            tf.math.multiply_no_nan(inputs, conv_mask),
            axis=1
        )
        return result


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

    # Charged constituents
    inputs_ch_numeric = keras.Input(
        shape=(features['ch']['max_size'], len(features['ch']['numeric'])),
        name='ch_numeric'
    )
    outputs_ch = _apply_dense_from_config(
        inputs_ch_numeric, model_config['ch'], name_prefix='ch_'
    )
    mask_ch = keras.Input(
        shape=(features['ch']['max_size'],), name='ch_mask'
    )
    outputs_ch = MaskedSum(name='ch_sum')(outputs_ch, mask=mask_ch)

    # Head
    inputs_global_numeric = keras.Input(
        shape=(len(features['global']['numeric']),),
        name='global_numeric'
    )
    inputs_head = Concatenate()([inputs_global_numeric, outputs_ch])
    outputs = _apply_dense_from_config(
        inputs_head, model_config['head'], name_prefix='head_'
    )

    # Automatically add the output unit
    outputs = Dense(1)(outputs)
    model = keras.Model(
        inputs=[inputs_global_numeric, mask_ch, inputs_ch_numeric],
        outputs=outputs
    )

    if config['loss'] == 'huber':
        loss = keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile('adam', loss=loss)
    return model


def _apply_dense_from_config(
    inputs: tf.Tensor, config: Mapping, name_prefix: str = ''
) -> tf.Tensor:
    """Wrapper around _apply_mlp and _apply_resnet.

    Args:
        inputs:  Inputs to the network fragment.
        config:  Configuration for the network fragment.
        name_prefix: Name prefix for layers.

    Return:
        Output of the fragment.
    """

    model_type = config['type']
    if model_type == 'mlp':
        outputs = _apply_mlp(
            inputs,
            num_units=config['num_units'],
            batch_norm=config.get('batch_norm', False),
            dropout=config.get('dropout', 0),
            name_prefix=name_prefix
        )
    elif model_type == 'resnet':
        outputs = _apply_resnet(
            inputs,
            num_units=config['num_units'],
            name_prefix=name_prefix
        )
    else:
        raise RuntimeError(f'Unknown model type "{model_type}".')
    return outputs


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

    Return:
        Output of the fragment.
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

    Return:
        Output of the fragment.

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
