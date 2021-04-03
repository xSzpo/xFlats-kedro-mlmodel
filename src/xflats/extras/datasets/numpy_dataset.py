from pathlib import PurePosixPath
from typing import Any, Dict

from kedro.io.core import (
    AbstractVersionedDataSet,
    get_filepath_str,
    get_protocol_and_path,
    Version
)

import fsspec
import numpy as np


class NumpySet(AbstractVersionedDataSet):
    """``NumpySet`` loads / save `npy` data from a given filepath as `numpy`.

    Example:
    ::

        >>> NumpySet(filepath='~/file/path.png')
    """

    def __init__(self, filepath: str, version: Version = None):
        """Creates a new instance of NumpySet to load / save data for given filepath.

        Args:
            filepath: The location of the `npy` file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> np.ndarray:
        """Loads data from the `npy` file.

        Returns:
            Data from the `npy` file as a numpy array
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            numpy_array = np.load(f, allow_pickle=True)
        return numpy_array

    def _save(self, data: np.ndarray) -> None:
        """Saves numpy array to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            np.save(f, data, allow_pickle=True)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
