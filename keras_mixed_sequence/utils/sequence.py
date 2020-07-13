from tensorflow.keras.utils import Sequence as KerasSequence
import numpy as np


class Sequence(KerasSequence):
    """Wrapper of Keras Sequence to handle some commonly used methods and properties."""

    def __init__(
        self,
        samples_number: int,
        batch_size: int,
        elapsed_epochs: int = 0
    ):
        """Return new Sequence object.

        Parameters
        --------------
        samples_number: int,
            Length of the sequence to be split into batches.
        batch_size: int,
            Batch size for the current Sequence.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.

        Returns
        --------------
        Return new Sequence object.
        """
        if samples_number < batch_size:
            raise ValueError((
                "Given sequence length ({}) "
                "is smaller than a single batch of size ({})."
            ).format(
                samples_number,
                batch_size
            ))
        if not isinstance(samples_number, int) or samples_number == 0:
            raise ValueError(
                "Given sequence length must be a strictly positive integer."
            )
        if not isinstance(batch_size, int) or batch_size == 0:
            raise ValueError(
                "Given batch size must be a strictly positive integer."
            )
        self._samples_number = samples_number
        self._batch_size = batch_size
        self._elapsed_epochs = elapsed_epochs

    def on_epoch_end(self):
        """Handled the on epoch end callback."""
        pass

    @property
    def batch_size(self) -> int:
        """Return batch size property of the sequence."""
        return self._batch_size

    def reset(self):
        """Reset sequence to before training was started."""
        self._elapsed_epochs = 0

    @property
    def elapsed_epochs(self):
        """Return elapsed epochs since training started."""
        return self._elapsed_epochs

    @property
    def samples_number(self):
        """Return total number of samples in sequence."""
        return self._samples_number

    def __len__(self) -> int:
        """Return length of Sequence."""
        return int(np.ceil(self.samples_number / self.batch_size))

    @property
    def steps_per_epoch(self):
        """Number of steps to execute on the sequence."""
        return len(self)