from .sequence import Sequence
import numpy as np


class VectorSequence(Sequence):
    """Wrapper of Keras Sequence to handle some commonly used methods and properties."""

    def __init__(
        self,
        vector: np.ndarray,
        batch_size: int,
        seed: int = 42,
        elapsed_epochs: int = 0
    ):
        """Return new Sequence object.

        Parameters
        -------------------------------------
        vector: np.ndarray,
            Numpy array with data to be split into batches.
        batch_size: int,
            Batch size for the current Sequence.
        seed: int = 42,
            Random seed to use for reproducibility.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.

        Returns
        -------------------------------------
        Return new Sequence object.
        """
        super().__init__(
            vector.shape[0],
            batch_size,
            elapsed_epochs
        )
        self._seed = seed
        self._vector = vector
        self._shuffled = vector.copy()

    def on_epoch_end(self):
        """Shuffle private numpy array on every epoch end."""
        state = np.random.RandomState(  # pylint: disable=no-member
            seed=self._seed + self._elapsed_epochs
        )
        self._elapsed_epochs += 1
        indices = np.arange(self.samples_number)
        state.shuffle(indices)
        self._shuffled = self._vector[indices]

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return batch corresponding to given index.

        Parameters
        ---------------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------------
        Return numpy array corresponding to given batch index.
        """
        return self._shuffled[idx * self.batch_size: (idx + 1) * self.batch_size]
