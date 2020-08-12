from collections.abc import Sequence
import math

from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Dense, Dropout
)


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
    if isinstance(model_config, str):
        model = keras.models.load_model(model_config)
        return model

    head_config = model_config['head']
    inputs = keras.Input(shape=(num_features, ), name='global_numeric')
    model_type = head_config['type']
    if model_type == 'plain':
        out = _apply_mlp(
            inputs,
            num_layers=head_config.get('num_layers', None),
            num_units=head_config['num_units'],
            batch_norm=head_config.get('batch_norm', False),
            dropout=head_config.get('dropout', 0)
        )
    elif model_type == 'resnet':
        out = _apply_resnet(
            inputs,
            num_layers=head_config['num_layers'],
            num_units=head_config['num_units']
        )
    else:
        raise RuntimeError(f'Unknown model type "{model_type}".')
    model = keras.Model(inputs=inputs, outputs=out)

    if config['loss'] == 'huber':
        loss = keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile('adam', loss=loss)
    return model


def _apply_mlp(
    inputs, num_layers=5, num_units=256, batch_norm=False, dropout=0
):
    """Apply an MLP fragment to given inputs.

    Args:
        inputs:  Inputs to the network fragment.
        num_layers:  Number of hidden layers.  Ignored if num_units is
            a sequence.
        num_units:  Number of units in each hidden layer.  Can be a
            sequence.  In that case it provides numbers of units in all
            hidden layers.
        batch_norm:  Whether to apply batch normalization.
        dropout:  Dropout rate.  Disabled if 0.
    """

    if not isinstance(num_units, Sequence):
        num_units = [num_units] * num_layers

    x = inputs
    for i, n in enumerate(num_units):
        x = Dense(
            n, kernel_initializer='he_uniform', use_bias=not batch_norm
        )(x)
        if batch_norm:
            x = BatchNormalization(scale=False)(x)
        x = Activation('relu')(x)
        if dropout > 0 and i != 0:
            x = Dropout(dropout)(x)
    x = Dense(1, kernel_initializer='he_uniform')(x)
    return x


def _apply_resnet(inputs, num_layers=5, num_units=256):
    """Apply a ResNet fragment to inputs.

    Args:
        inputs:  Inputs for the network fragment.
        num_layers:  Number of ResNet blocks.  Each block includes two
            sublayers with non-linearities.
        num_units:  Number of units in each hidden layer.
    """

    # The first layer is special in that it includes a projection for
    # the inputs in order to match the dimensions
    x = Dense(
        num_units, kernel_initializer='he_uniform', use_bias=False
    )(inputs)
    y = Dense(num_units, kernel_initializer='he_uniform')(x)
    y = Activation('relu')(y)
    y = Dense(num_units, kernel_initializer='he_uniform')(y)
    y = Add()([x, y])
    x = Activation('relu')(y)

    for _ in range(num_layers - 1):
        y = Dense(num_units, kernel_initializer='he_uniform')(x)
        y = Activation('relu')(y)
        y = Dense(num_units, kernel_initializer='he_uniform')(y)
        y = Add()([x, y])
        x = Activation('relu')(y)

    output = Dense(1, kernel_initializer='he_uniform')(x)
    return output
