"""Conversion of LabOne Q results to qibolab ones."""

import numpy as np

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import Result

from .sweep import Axes

__all__ = []


def parse(data: np.ndarray, options: ExecutionParameters, axes: Axes) -> Result:
    """Convert the integrated data of one acquisition to the qibolab shape.

    LabOne Q returns complex data with axes (near-time sweeps, shots, real-time
    sweeps), while qibolab expects (shots, sweeps in their order, i/q). ``axes``
    gives the qibolab sweeper of each LabOne Q axis, as built by
    :func:`.sweep.split`; the shots axis is there only when averaging in single shot.
    """
    shots = options.averaging_mode is AveragingMode.SINGLESHOT
    source = [axis for axis in axes if shots or axis is not None]
    data = np.moveaxis(
        data,
        list(range(len(source))),
        [0 if axis is None else axis + int(shots) for axis in source],
    )
    return np.stack([np.real(data), np.imag(data)], axis=-1)
