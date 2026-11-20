"""Conversion of LabOne Q results to qibolab ones."""

import numpy as np

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result

from .sweep import Axes

__all__ = []


def parse(data: np.ndarray, options: ExecutionParameters, axes: Axes) -> Result:
    """Convert the data of one acquisition to the qibolab shape.

    LabOne Q returns complex data with axes (near-time sweeps, shots, real-time
    sweeps), while qibolab expects (shots, sweeps in their order, i/q). ``axes``
    gives the qibolab sweeper of each LabOne Q axis, as built by
    :func:`.sweep.split`; the shots axis is there only when averaging in single shot.
    Discriminated data has the state in the real part, and no i/q axis: states in
    single shot, probabilities of the excited state when averaged.
    """
    shots = options.averaging_mode is AveragingMode.SINGLESHOT
    source = [axis for axis in axes if shots or axis is not None]
    data = np.moveaxis(
        data,
        list(range(len(source))),
        [0 if axis is None else axis + int(shots) for axis in source],
    )
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        return np.real(data)
    return np.stack([np.real(data), np.imag(data)], axis=-1)
