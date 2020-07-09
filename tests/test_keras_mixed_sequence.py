import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras_mixed_sequence import MixedSequence
from keras_mixed_sequence.utils import NumpySequence


def build_model():
    inputs = Input(shape=(10,))

    output1 = Dense(
        units=10,
        activation="relu",
        name="output1"
    )(inputs)
    output2 = Dense(
        units=10,
        activation="relu",
        name="output2"
    )(inputs)

    model = Model(
        inputs=inputs,
        outputs=[output1, output2],
        name="my_model"
    )

    model.compile(
        optimizer="nadam",
        loss="MSE"
    )

    return model


def test_keras_mixed_sequence():
    model = build_model()
    batch_size = 32
    sequence = MixedSequence(
        np.random.randint(2, size=(100, 10)),
        {
            "output1": NumpySequence(
                np.random.randint(2, size=(100, 10)),
                batch_size=batch_size
            ),
            "output2": np.random.randint(2, size=(100, 10))
        },
        batch_size=batch_size
    )
    model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )

    model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
