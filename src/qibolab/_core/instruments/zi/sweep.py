"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sweeper import ParallelSweepers, Parameter

__all__ = []

Amplitudes = dict[PulseId, SweepParameter]
"""Swept amplitude of each pulse, by pulse id."""


def near_time(
    sweepers: list[ParallelSweepers],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    amplitudes: Amplitudes,
) -> list[list[SweepParameter]]:
    """Sweep parameters of each group of parallel sweepers, all in near time.

    Supported sweeps:

    - frequency of software-modulated (probe) channels: the parameter replaces the
      frequency of the channel's oscillator in ``calibration``;
    - amplitude of pulses: the parameter is recorded in ``amplitudes``, to be used
      when the pulse is played.
    """
    groups = []
    for group in sweepers:
        parameters = []
        for sweeper in group:
            if sweeper.parameter is Parameter.frequency:
                for id in sweeper.channels:
                    oscillator = calibration[id].oscillator
                    if oscillator.modulation_type is not ModulationType.SOFTWARE:
                        raise NotImplementedError(
                            f"Sweeping the frequency of '{id}' is not supported."
                        )
                    lo = configs[channels[id].lo].frequency
                    oscillator.frequency = SweepParameter(values=sweeper.values - lo)
                    parameters.append(oscillator.frequency)
            elif sweeper.parameter is Parameter.amplitude:
                for pulse in sweeper.pulses:
                    amplitudes[pulse.id] = SweepParameter(values=sweeper.values)
                    parameters.append(amplitudes[pulse.id])
            else:
                raise NotImplementedError(
                    f"Sweeping {sweeper.parameter.name} is not supported."
                )
        groups.append(parameters)
    return groups
