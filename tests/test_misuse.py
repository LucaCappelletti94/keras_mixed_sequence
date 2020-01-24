import numpy as np
from keras_mixed_sequence import MixedSequence
from keras_mixed_sequence.utils import NumpySequence
import pytest


def test_misuse():
    with pytest.raises(ValueError):
        MixedSequence(
            NumpySequence(np.random.randint(2, size=(100, 10)), batch_size=32),
            np.random.randint(2, size=(60, 10)),
            batch_size=32
        )

    with pytest.raises(ValueError):
        MixedSequence(
            np.random.randint(2, size=(60, 10)),
            np.random.randint(2, size=(60, 10)),
            batch_size=32
        )
