from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow import keras

import mljec


def test_sum_layer():
    inputs = tf.RaggedTensor.from_row_lengths(
        values=[[1] * 4, [2] * 4, [10] * 4],
        row_lengths=[2, 1, 0]
    )
    result = mljec.model.Sum()(inputs)
    expected_result = np.array(
        [[3] * 4, [10] * 4, [0] * 4],
        dtype=np.float32
    )
    assert np.array_equal(result, expected_result)


def test_mlp():
    inputs = keras.Input(shape=(10,))
    outputs = mljec.model._apply_mlp(
        inputs, [10] * 3, batch_norm=True, dropout=0.1
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    assert model.count_params() == 390


def test_resnet():
    inputs = keras.Input(shape=(10,))
    outputs = mljec.model._apply_resnet(
        inputs, [10] * 3 + [5]
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    assert model.count_params() == 435


def test_deep_set():
    config = {
        'embeddings': {
            'ch_id': 3,
            'ch_pv_ass': 2
        },
        'type': 'mlp',
        'num_units': [10, 10]
    }
    cardinalities = {'ch_id': 6, 'ch_pv_ass': 6}
    inputs_numerical = keras.Input(shape=(None, 10), ragged=True)
    inputs_categorical = OrderedDict()
    for feature in cardinalities:
        inputs_categorical[feature] = keras.Input(shape=(None,), ragged=True)
    outputs = mljec.model._apply_deep_set(
        inputs_numerical, inputs_categorical, config, cardinalities, 'deep_set'
    )
    model = keras.Model(
        inputs=[inputs_numerical] + list(inputs_categorical.values()),
        outputs=outputs
    )
    assert model.count_params() == 300
