"""Conversion of qibolab sequences to LabOne Q operations."""

from functools import cache

import numpy as np
from laboneq.simple import Experiment, Section, pulse_library
from pydantic import TypeAdapter

from qibolab._core.components import Channel, Config, IqChannel
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import (
    Align,
    Delay,
    Envelope,
    Pulse,
    PulseId,
    Readout,
    Rectangular,
)
from qibolab._core.sequence import PulseSequence

from .sweep import NS, Swept

__all__ = []


@pulse_library.register_pulse_functional
def qibolab_envelope(x, length, sampling_rate, envelope: dict, **_):
    """Samples of the qibolab ``envelope`` at the points ``x``, in [-1, 1).

    The envelope is sampled at the pulse's actual length, so that it can be swept.
    """
    shape = TypeAdapter(Envelope).validate_python(envelope)
    samples = round(length * sampling_rate)
    index = np.rint((x + 1) / 2 * samples).astype(int)
    return (shape.i(samples) + 1j * shape.q(samples))[index]


@cache
def waveform(envelope: Envelope, duration: float):
    """Unit-amplitude LabOne Q pulse with the shape of a qibolab envelope.

    Equal envelopes give the same object, since an SHFQA output can only hold one
    waveform.
    """
    return qibolab_envelope(
        length=duration * NS, envelope=envelope.model_dump(mode="json")
    )


def play(
    experiment: Experiment,
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    swept: Swept,
) -> None:
    """Add the operations of ``sequence`` to the current section of ``experiment``.

    The operations of each channel go into a section of their own, and an
    :class:`Align` starts new sections on its channels, played after the previous
    ones of the same channels.

    A :class:`Readout` plays its probe pulse and acquires at the same time, integrating
    with unit weights (custom kernels are not supported). A delay on an acquisition
    channel also delays its probe channel, to keep them aligned. Swept arguments are
    taken from ``swept``, by the id of the operation.
    """
    sections: dict[ChannelId, Section] = {}
    after: dict[ChannelId, list[str]] = {}
    aligns: dict[PulseId, list[str]] = {}

    def section(id: ChannelId) -> Section:
        """Current section of the channel ``id``."""
        if id not in sections:
            sections[id] = Section(play_after=after.pop(id, None))
            experiment.add(sections[id])
        return sections[id]

    for id, op in sequence:
        if isinstance(op, Align):
            if op.id not in aligns:
                aligns[op.id] = [
                    section(ch).uid for ch in sequence.pulse_channels(op.id)
                ]
            after[id] = aligns[op.id]
            del sections[id]
            continue
        channel = channels[id]
        probe = channel.iqout(id)
        kwargs = swept.get(op.id, {})
        if isinstance(op, Delay):
            for signal in {id, probe} - {None}:
                section(id).delay(signal, **({"time": op.duration * NS} | kwargs))
        elif isinstance(op, Pulse) and isinstance(channel, IqChannel):
            section(id).play(
                id,
                pulse=waveform(op.envelope, op.duration),
                phase=op.relative_phase,
                **({"amplitude": op.amplitude} | kwargs),
            )
        elif isinstance(op, Readout) and configs[id].kernel is None:
            section(id).play(
                probe,
                pulse=waveform(op.probe.envelope, op.probe.duration),
                phase=op.probe.relative_phase,
                **({"amplitude": op.probe.amplitude} | kwargs),
            )
            section(id).acquire(
                id,
                handle=str(op.id),
                kernel=waveform(Rectangular(), op.acquisition.duration),
            )
        else:
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{id}' is not supported."
            )
