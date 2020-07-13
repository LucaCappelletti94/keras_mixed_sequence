from keras_mixed_sequence import MixedSequence, VectorSequence
import numpy as np


def test_simple_determinism():
    """Test to check that the extraction of the batches is deterministic."""
    classes = 10
    number = 100000
    epochs = 100
    batch_size = 10000

    x = np.arange(0, number, dtype=np.int64)
    y = np.random.randint(0, classes, size=number)

    ms = MixedSequence(
        VectorSequence(x, batch_size),
        VectorSequence(y, batch_size)
    )

    ms2 = MixedSequence(
        VectorSequence(x, batch_size),
        VectorSequence(y, batch_size)
    )

    for epoch in range(epochs, desc="Epochs", leave=False):
        for step in range(ms.steps_per_epoch, desc="Batches", leave=False):
            xi, yi = ms[step]
            xj, yj = ms2[step]
            if epoch == 0:
                # The first epochs they must be aligned
                assert (xi == xj).all()
                assert (yi == yj).all()
            else:
                # Afterwards, since the ms2 is not shuffled, they must not be
                # anymore. Or at least, is very unlikely.
                assert (xi != xj).any()
                assert (yi != yj).any()
            assert (y[xi] == yi).all()

        ms.on_epoch_end()
