"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

import numpy as np
from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Delay, PulseId, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

__all__ = []

NS = 1e-9
"""Qibolab times are in ns, LabOne Q ones in s."""

Swept = dict[PulseId, dict[str, SweepParameter]]
"""Swept arguments of each operation, by its id and the LabOne Q keyword."""

Axes = list[int | None]
"""Qibolab sweeper of each axis of the LabOne Q results, ``None`` for the shots."""


def _channels_of(sequences: list[PulseSequence]) -> dict[PulseId, ChannelId]:
    """Channel each operation of the sequences is played on, by its id."""
    return {op.id: id for sequence in sequences for id, op in sequence}


def _real_time(id: ChannelId, calibration: Calibration) -> bool:
    """Whether a sweep of the channel ``id`` runs in real time.

    It does if the channel's oscillator is modulated by the hardware (drive), and
    not if it is modulated by the software (probe and acquisition): an SHFQA
    generator cannot be swept in real time in integration mode.
    """
    oscillator = calibration[id].oscillator
    return (
        oscillator is not None and oscillator.modulation_type is ModulationType.HARDWARE
    )


def _group(
    group: ParallelSweepers,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    swept: Swept,
    pulse_channels: dict[PulseId, ChannelId],
) -> tuple[list[SweepParameter], bool]:
    """Sweep parameters of one group of parallel sweepers, and their execution type.

    Supported sweeps, each on the channel's own time scale (:func:`_real_time`):

    - frequency of a channel: the parameter replaces the frequency of the channel's
      oscillator in ``calibration``;
    - offset of a DC channel: the parameter replaces its ``voltage_offset`` in
      ``calibration``, so the values are absolute. A DC channel has no oscillator, so
      it is swept in near time;
    - amplitude and duration of pulses, and duration of delays: the parameter is
      recorded in ``swept``, to be used when the operation is played. The channel of a
      pulse is the one it is played on in the sequences, which for a :class:`Readout`
      is its acquisition channel. Delays are always swept in real time.
    """
    parameters = []
    real_time = set()
    for sweeper in group:
        if sweeper.parameter is Parameter.frequency:
            for id in sweeper.channels:
                oscillator = calibration[id].oscillator
                lo = configs[channels[id].lo].frequency
                oscillator.frequency = SweepParameter(values=sweeper.values - lo)
                parameters.append(oscillator.frequency)
                real_time.add(_real_time(id, calibration))
        elif sweeper.parameter is Parameter.offset:
            for id in sweeper.channels:
                calibration[id].voltage_offset = SweepParameter(values=sweeper.values)
                parameters.append(calibration[id].voltage_offset)
                real_time.add(_real_time(id, calibration))
        elif sweeper.parameter is Parameter.amplitude:
            for pulse in {p.id: p for p in sweeper.pulses}.values():
                swept.setdefault(pulse.id, {})["amplitude"] = SweepParameter(
                    values=sweeper.values
                )
                parameters.append(swept[pulse.id]["amplitude"])
                real_time.add(_real_time(pulse_channels[pulse.id], calibration))
        elif sweeper.parameter is Parameter.duration:
            for pulse in {p.id: p for p in sweeper.pulses}.values():
                if isinstance(pulse, Readout):
                    raise NotImplementedError("Sweeping a readout duration.")
                delay = isinstance(pulse, Delay)
                key = "time" if delay else "length"
                swept.setdefault(pulse.id, {})[key] = SweepParameter(
                    values=sweeper.values * NS
                )
                parameters.append(swept[pulse.id][key])
                real_time.add(
                    delay or _real_time(pulse_channels[pulse.id], calibration)
                )
        else:
            raise NotImplementedError(
                f"Sweeping {sweeper.parameter.name} is not supported."
            )
    if len(real_time) > 1:
        raise NotImplementedError(
            "Sweeping in parallel in near time and in real time is not supported."
        )
    return parameters, real_time == {True}


def split(
    sweepers: list[ParallelSweepers],
    sequences: list[PulseSequence],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    swept: Swept,
) -> tuple[list[list[SweepParameter]], list[list[SweepParameter]], Axes]:
    """Split the sweepers in near-time and real-time groups, keeping their order.

    Each group of parallel sweepers becomes one list of sweep parameters, to be
    opened as a single LabOne Q sweep. See :func:`_group` for the supported sweeps
    and for which of them run in real time; ``sequences`` are the ones to be played,
    which tell on which channel each swept pulse is.

    The last element maps the axes of the LabOne Q results, which are the near-time
    sweeps, then the shots, then the real-time ones, back to the qibolab sweepers.
    """
    near: list[list[SweepParameter]] = []
    real: list[list[SweepParameter]] = []
    near_axes: Axes = []
    real_axes: Axes = []
    pulse_channels = _channels_of(sequences)
    for index, group in enumerate(sweepers):
        parameters, real_time = _group(
            group, channels, configs, calibration, swept, pulse_channels
        )
        # LabOne Q cannot compile an unevenly spaced frequency sweep nested in a
        # real-time sweep ("c_freq_osc_0 already declared")
        if real and any(
            s.parameter is Parameter.frequency
            and abs(np.diff(s.values, 2)).max(initial=0) > 1e-4
            for s in group
        ):
            real_time = False
        if real_time:
            real.append(parameters)
            real_axes.append(index)
        else:
            near.append(parameters)
            near_axes.append(index)
    return near, real, near_axes + [None] + real_axes
