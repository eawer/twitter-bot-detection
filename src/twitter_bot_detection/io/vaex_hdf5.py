import vaex

from typing import Any, Dict, List

from kedro.io import AbstractDataSet


class VaexHDF5DataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> vaex.hdf5.dataset.Hdf5MemoryMapped:
        return vaex.open(self._filepath)

    def _save(self, data: vaex.hdf5.dataset.Hdf5MemoryMapped) -> None:
        data.export_hdf5(path=self._filepath)

    def _describe(self) -> Dict[str, Any]:
        pass
