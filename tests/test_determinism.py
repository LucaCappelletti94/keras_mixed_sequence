from keras_mixed_sequence import MixedSequence
import numpy as np


def test_simple_determinism():
    classes = 10
    number = 100000
    epochs = 100
    x = np.arange(0, number, dtype=np.int64)
    y = np.random.randint(0, classes, size=number)

    ms = MixedSequence(x, y, batch_size=100)

    for _ in range(epochs):
        for step in range(ms.steps_per_epoch):
            xi, yi = ms[step]
            assert (y[xi.astype(int)] == yi).all()
        ms.on_epoch_end()
