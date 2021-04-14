from keras_mixed_sequence import Sequence
import pytest


def test_sequence():
    with pytest.raises(ValueError):
        Sequence(0, 10, 0)

    with pytest.raises(ValueError):
        Sequence(10, 0, 0)

    with pytest.raises(ValueError):
        Sequence(10, 10, -1)


def test_sequence_reset():
    seq = Sequence(50, 10, 5)
    seq.reset()
    assert seq.elapsed_epochs == 0
