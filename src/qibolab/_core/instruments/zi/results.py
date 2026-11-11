"""Conversion of LabOne Q results to qibolab ones."""

import numpy as np

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import Result

__all__ = []


def parse(data: np.ndarray, options: ExecutionParameters, near_time: int) -> Result:
    """Convert the integrated data of one acquisition to the qibolab shape.

    LabOne Q returns complex data with axes (near-time sweeps, shots), while qibolab
    expects (shots, sweeps, i/q). ``near_time`` is the number of near-time sweeps.
    """
    if options.averaging_mode is AveragingMode.SINGLESHOT:
        data = np.moveaxis(data, near_time, 0)
    return np.stack([np.real(data), np.imag(data)], axis=-1)
