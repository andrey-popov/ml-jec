import math
import os
from typing import Callable, List, Mapping, OrderedDict, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Dense, Dropout,
    Embedding, TimeDistributed
)
import tensorflow_addons as tfa

from .data import MaybeRaggedTensor
from .util import Features


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
        model = keras.models.load_model(
            model_config, custom_objects={
                'loss_fn': _create_loss(config['loss'])
            }
        )

        return model

    features = Features(config['features'])
    inputs_all = []

    # Constituents of different types
    constituent_types = [
        key for key in sorted(model_config.keys())  # Ensure order
        if key not in {'head', 'load_weights'}
    ]
    outputs_constituents = []
    for constituent_type in constituent_types:
        inputs_numerical = keras.Input(
            shape=(None, len(features.numerical(constituent_type))),
            ragged=True, name=f'{constituent_type}_numerical'
        )
        inputs_categorical = OrderedDict()
        for feature in features.categorical(constituent_type):
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
        shape=(len(features.numerical('global')),),
        name='global_numerical'
    )
    inputs_global_categorical = OrderedDict()
    for feature in features.categorical('global'):
        inputs_global_categorical[feature] = keras.Input(
            shape=(None,), name=feature
        )
    embeddings_global = {
        feature: Embedding(
            cardinalities[feature],
            model_config['head']['embeddings'][feature],
            name=feature + '_embeddings'
        )(inputs)
        for feature, inputs in inputs_global_categorical.items()
    }
    inputs_all.append(inputs_global_numerical)
    inputs_all.extend(inputs_global_categorical.values())
    inputs_head = Concatenate(name='head_concatenate')(
        [inputs_global_numerical]
        + [
            embeddings_global[feature]
            for feature in inputs_global_categorical.values()
        ]
        + outputs_constituents
    )
    outputs = _apply_dense_from_config(
        inputs_head, model_config['head'], name_prefix='head_'
    )

    outputs = Dense(1, name='head_dense_output')(outputs)  # Output unit
    model = keras.Model(inputs=inputs_all, outputs=outputs, name='full')

    model.compile(
        optimizer=_create_optimizer(config.get('optimizer', None)),
        loss=_create_loss(config['loss'])
    )
    if 'load_weights' in model_config:
        # Normally, a saved model should be loaded
        # keras.models.load_model at the beginning of thsi function.
        # However, this is currently not supported for models that use
        # ragged tensors [1].  As a workaround, construct the model anew
        # and then load saved weights.  The path to weights would
        # usually be "{model_directory}/variables/variables", with the
        # ".index" file extension stripped off.  This doesn't restore
        # the state of the optimizer.
        # [1] https://github.com/tensorflow/tensorflow/issues/41034
        model.load_weights(model_config['load_weights'])
    return model


def summarize_model(
    model: keras.Model, fig_dir: Union[str, None] = None
) -> None:
    """Print an optionally plot a model.

    Do this for the main model and all submodels wrapped in
    TimeDistributed layers.

    Args:
        model:  Model as created by build_model function.
        fig_dir:  Directory for plots of the model.  If None, do not
            produce the plots.
    """

    submodels = []
    for layer in model.layers:
        if isinstance(layer, TimeDistributed):
            submodels.append(layer.layer)

    for submodel in submodels:
        submodel.summary()
    model.summary()

    if fig_dir is not None:
        for submodel in submodels:
            keras.utils.plot_model(
                submodel, os.path.join(fig_dir, f'model_{submodel.name}.png'),
                dpi=300
            )
        keras.utils.plot_model(
            model, os.path.join(fig_dir, 'model_full.png'), dpi=300
        )


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


def _create_loss(
    loss_name: str
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Create loss function."""

    if loss_name not in {'mae'}:
        raise RuntimeError(f'Loss {loss_name} is not implemented.')

    def loss_fn(y_true, y_pred):
        base_loss = tf.math.abs(y_pred - y_true)
        # Skip jets with extreme values for the target
        mask = tf.math.logical_and(y_true > -1., y_true < 1.)
        return tf.math.reduce_mean(base_loss * tf.cast(mask, tf.float32))

    return loss_fn


def _create_optimizer(
    config: Union[None, Mapping] = None
) -> keras.optimizers.Optimizer:
    """Construct an optimizer from configuration."""

    if config is None:
        return keras.optimizers.Adam()

    algorithm = config['algorithm'].lower()
    if algorithm == 'adam':
        return keras.optimizers.Adam()
    elif algorithm == 'sgd':
        return keras.optimizers.SGD(
            momentum=config.get('momentum', 0.),
            nesterov=config.get('nesterov', False)
        )
    elif algorithm == 'radam':
        return tfa.optimizers.RectifiedAdam()
    else:
        raise RuntimeError(
            'Unsupported optimizer "{}".'.format(config['algorithm'])
        )
