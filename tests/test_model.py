import math

import numpy as np
from tensorflow import keras

import mljec


def test_masked_sum():
    inputs = [
        [[1] * 4, [2] * 4, [math.nan] * 4],
        [[10] * 4, [math.nan] * 4, [math.nan] * 4],
        [[math.nan] * 4] * 3
    ]
    inputs = np.array(inputs, dtype=np.float32)
    mask = [
        [True, True, False],
        [True, False, False],
        [False, False, False]
    ]
    result = mljec.model.MaskedSum()(inputs, mask)
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
