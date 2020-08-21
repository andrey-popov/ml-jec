import math
from typing import List, Mapping, OrderedDict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Dense, Dropout,
    Embedding, TimeDistributed
)

from .data import MaybeRaggedTensor


class Sum(keras.layers.Layer):
    """Layer that sums input tensor along an axis."""

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: MaybeRaggedTensor) -> tf.Tensor:
        return tf.math.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config


def build_model(
    config: Mapping, cardinalities: Mapping[str, int]
) -> keras.Model:
    """Construct model specified in the configuration.

    Also create optimizer and set the loss function.

    Args:
        config:  Dictionary representing configuration file.
        cardinalities:  Cardinalities of categorical features (needed to
            construct their embeddings).

    Return:
        Compiled model.
    """

    model_config = config['model']
    if isinstance(model_config, str):
        model = keras.models.load_model(model_config)
        return model

    features = config['features']
    inputs_all = []

    # Constituents of different types
    constituent_types = sorted(model_config.keys())  # Ensure order
    constituent_types.remove('head')
    outputs_constituents = []
    for constituent_type in constituent_types:
        inputs_numerical = keras.Input(
            shape=(None, len(features[constituent_type]['numerical'])),
            ragged=True, name=f'{constituent_type}_numerical'
        )
        inputs_categorical = OrderedDict()
        for feature in features[constituent_type]['categorical']:
            inputs_categorical[feature] = keras.Input(
                shape=(None,), ragged=True, name=feature
            )
        inputs_all.append(inputs_numerical)
        inputs_all.extend(inputs_categorical.values())

        outputs = _apply_deep_set(
            inputs_numerical, inputs_categorical,
            model_config[constituent_type], cardinalities, constituent_type
        )
        outputs_constituents.append(outputs)

    # Head
    inputs_global_numerical = keras.Input(
        shape=(len(features['global']['numerical']),),
        name='global_numerical'
    )
    inputs_all.append(inputs_global_numerical)
    inputs_head = Concatenate(name='head_concatenate')(
        [inputs_global_numerical] + outputs_constituents
    )
    outputs = _apply_dense_from_config(
        inputs_head, model_config['head'], name_prefix='head_'
    )

    outputs = Dense(1)(outputs)  # Output unit
    model = keras.Model(inputs=inputs_all, outputs=outputs)

    if config['loss'] == 'huber':
        loss = keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile('adam', loss=loss)
    return model


def _apply_deep_set(
    inputs_numerical: tf.RaggedTensor,
    inputs_categorical: OrderedDict[str, tf.RaggedTensor],
    config: Mapping, cardinalities: Mapping[str, int], name: str
) -> tf.Tensor:
    """Apply a DeepSet-like fragment to constituents of certain type.

    Apply the same subnetwork to each constituent and then sum its
    outputs over constituents.

    Args:
        inputs_numerical:  Numerical features.
        inputs_categorical:  Categorical features.  The order fixes the
            order of inputs to the dense subnetwork.
        config:  Fragment of the model configuration that describes this
            block.  Must include information about embeddings for
            categorical features and configuration for the dense
            subnetwork.  This mapping is also forwarded to
            _apply_dense_from_config.
        cardinalities:  Cardinalities of categorical features.  Needed
            to construct the embeddings.
        name:  Name fragment for layers.

    Return:
        Output of the fragment.
    """

    # Apply embeddings to categorical features and concatenate all
    # features for each constituent forming dense blocks
    embeddings = {
        feature: Embedding(
            cardinalities[feature], config['embeddings'][feature],
            name=feature + '_embedding'
        )(inputs)
        for feature, inputs in inputs_categorical.items()
    }
    inputs_all = (
        [inputs_numerical]
        + [embeddings[feature] for feature in inputs_categorical.keys()]
    )
    if len(inputs_all) > 1:
        inputs_concat = Concatenate(name=f'{name}_concatenate')(inputs_all)
    else:
        # Concatenate doesn't work with a single input
        inputs_concat = inputs_all[0]

    # Construct per-constituent subnetwork as a submodel and apply it to
    # each constituent or slice using TimeDistributed layer
    inputs_slice = keras.Input(shape=(inputs_concat.shape[-1],))
    outputs_slice = _apply_dense_from_config(
        inputs_slice, config, name_prefix=f'{name}_'
    )
    submodel_slice = keras.Model(
        inputs=inputs_slice, outputs=outputs_slice,
        name=name
    )
    outputs = TimeDistributed(
        submodel_slice, name=f'{name}_distributed'
    )(inputs_concat)

    # Sum over constituents
    outputs = Sum(name=f'{name}_sum')(outputs)

    return outputs


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
