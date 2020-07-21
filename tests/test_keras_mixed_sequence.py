import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras_mixed_sequence import MixedSequence, VectorSequence
import pytest


def build_model():
    input_layer = Input(shape=(10,))
    output1 = Dense(units=10, activation="relu", name="output1")(input_layer)
    output2 = Dense(units=20, activation="relu", name="output2")(input_layer)
    model = Model(inputs=input_layer, outputs=[output1, output2])
    model.compile(loss="mse", optimizer="nadam")
    return model


def test_keras_mixed_sequence():
    model = build_model()
    batch_size = 32
    sequence = MixedSequence(
        VectorSequence(np.random.randint(2, size=(100, 10)), batch_size),
        {
            "output1": VectorSequence(np.random.randint(2, size=(100, 10)), batch_size),
            "output2": VectorSequence(np.random.randint(2, size=(100, 20)), batch_size)
        }
    )
    model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )


def test_illegal_parameters_keras_mixed_sequence():
    with pytest.raises(ValueError):
        MixedSequence(
            VectorSequence(np.random.randint(2, size=(100, 10)), 20),
            VectorSequence(np.random.randint(2, size=(100, 10)), 50)
        )

    with pytest.raises(ValueError):
        MixedSequence(
            VectorSequence(np.random.randint(
                2, size=(100, 10)), 50, elapsed_epochs=50),
            VectorSequence(np.random.randint(2, size=(100, 10)), 50)
        )

    with pytest.raises(ValueError):
        MixedSequence(
            VectorSequence(np.random.randint(2, size=(60, 10)), 50),
            VectorSequence(np.random.randint(2, size=(100, 10)), 50)
        )

    with pytest.raises(ValueError):
        VectorSequence(np.random.randint(2, size=(60, 10)), 50)[10000]
