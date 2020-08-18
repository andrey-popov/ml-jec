from tensorflow import keras

import mljec


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
