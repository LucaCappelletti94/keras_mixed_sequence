"""Testing if proper exceptions are raised when wrong parameters are passed."""
import numpy as np
import pytest
from keras_mixed_sequence import MixedSequence
from keras_mixed_sequence.utils import NumpySequence


def test_misuse():
    """Testing if proper exceptions are raised when wrong parameters are passed."""
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
