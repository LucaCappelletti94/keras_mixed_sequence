from keras_mixed_sequence import VectorSequence, MixedSequence
import numpy as np


def test_change_batch_size():
    batch_size = 512
    sequence = MixedSequence(
        VectorSequence(np.empty(4096), batch_size=batch_size),
        VectorSequence(np.empty(4096), batch_size=batch_size),
    )
    print(sequence[0])
    assert sequence.batch_size == batch_size
    new_batch_size = 32
    sequence.batch_size = new_batch_size
    assert sequence.batch_size == 32
    sequence[sequence.steps_per_epoch-1]
