"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.sweeper import ParallelSweepers, Parameter

__all__ = []


def near_time(
    sweepers: list[ParallelSweepers],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
) -> list[list[SweepParameter]]:
    """Sweep parameters of each group of parallel sweepers, all in near time.

    Only frequency sweeps of software-modulated (probe) channels are supported: the
    parameter replaces the frequency of the channel's oscillator in ``calibration``.
    """
    groups = []
    for group in sweepers:
        parameters = []
        for sweeper in group:
            if sweeper.parameter is not Parameter.frequency:
                raise NotImplementedError(
                    f"Sweeping {sweeper.parameter.name} is not supported."
                )
            for id in sweeper.channels:
                oscillator = calibration[id].oscillator
                if oscillator.modulation_type is not ModulationType.SOFTWARE:
                    raise NotImplementedError(
                        f"Sweeping the frequency of '{id}' is not supported."
                    )
                lo = configs[channels[id].lo].frequency
                oscillator.frequency = SweepParameter(values=sweeper.values - lo)
                parameters.append(oscillator.frequency)
        groups.append(parameters)
    return groups
