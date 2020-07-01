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
    if isinstance(model_config, str):
        model = tf.keras.models.load_model(model_config)
        return model

    model_type = model_config['type']
    if model_type == 'plain':
        model = _build_model_plain(
            num_layers=model_config.get('num_layers', None),
            num_units=model_config['num_units'],
            batch_norm=model_config.get('batch_norm', False),
            dropout=model_config.get('dropout', 0),
            num_features=num_features
        )
    elif model_type == 'resnet':
        model = _build_model_resnet(
            num_layers=model_config['num_layers'],
            num_units=model_config['num_units'],
            num_features=num_features
        )
    else:
        raise RuntimeError(f'Unknown model type "{model_type}".')

    if config['loss'] == 'huber':
        loss = tf.keras.losses.Huber(math.log(2))
    else:
        loss = config['loss']

    model.compile('adam', loss=loss)
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
    model.add(layers.Dense(1, kernel_initializer='he_uniform'))
    return model


def _build_model_resnet(
    num_layers=5, num_units=256, num_features=None
):
    """Construct a ResNet.

    Args:
        num_layers:  Number of ResNet blocks.  Each block includes two
            sublayers with non-linearities.
        num_units:  Number of units in each hidden layer.
        num_features:  Optional number of input features.
    """

    inputs = tf.keras.Input(shape=(num_features,))

    # The first layer is special in that it includes a projection for
    # the inputs in order to match the dimensions
    x = layers.Dense(
        num_units, kernel_initializer='he_uniform', use_bias=False
    )(inputs)
    y = layers.Dense(num_units, kernel_initializer='he_uniform')(x)
    y = layers.Activation('relu')(y)
    y = layers.Dense(num_units, kernel_initializer='he_uniform')(y)
    y = layers.Add()([x, y])
    x = layers.Activation('relu')(y)

    for _ in range(num_layers - 1):
        y = layers.Dense(num_units, kernel_initializer='he_uniform')(x)
        y = layers.Activation('relu')(y)
        y = layers.Dense(num_units, kernel_initializer='he_uniform')(y)
        y = layers.Add()([x, y])
        x = layers.Activation('relu')(y)

    output = layers.Dense(1, kernel_initializer='he_uniform')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
