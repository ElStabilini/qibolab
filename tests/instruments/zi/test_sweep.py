"""Tests the split of the qibolab sweepers in near-time and real-time groups."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.instruments.zi.controller import calibration
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.pulses import Delay, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import INSTRUMENTS
from .test_calibration import DRIVE_LO, PROBE_LO, channels, configs  # noqa: F401

IDS = ["q0/drive", "q0/probe", "q0/acquisition"]


def sequence() -> tuple[PulseSequence, Pulse, Readout]:
    """A drive pulse and a delayed readout on ``q0``, as the rabi protocols."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    readout = Readout.from_probe(
        Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    )
    return (
        PulseSequence(
            [
                ("q0/drive", drive),
                ("q0/acquisition", Delay(duration=40)),
                ("q0/acquisition", readout),
            ]
        ),
        drive,
        readout,
    )


def test_split(channels, configs):  # noqa: F811
    """Sweeps on the drive go to real time, the ones on the probe to near time."""
    calib = calibration(IDS, channels, INSTRUMENTS, configs)
    _, pulse, readout = sequence()
    drive = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.30e9, 6.31e9]),
        channels=["q0/drive"],
    )
    amplitude = Sweeper(
        parameter=Parameter.amplitude, values=np.array([0.1, 0.2]), pulses=[readout]
    )
    probe = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([7.0e9]),
        channels=["q0/probe"],
    )
    amplitudes = {}
    # the pulse is played on the probe channel, so its amplitude sweep is near time
    probe_sequence = PulseSequence([("q0/probe", pulse)])

    near, real, axes = split(
        [[drive], [amplitude], [probe]],
        [probe_sequence],
        channels,
        configs,
        calib,
        amplitudes,
    )

    # LabOne Q returns the near-time sweeps (1 and 2), the shots, then the drive (0)
    assert axes == [1, 2, None, 0]
    assert len(near) == 2 and len(real) == 1
    # frequencies reach LabOne Q as intermediate frequencies
    assert real[0][0].values == pytest.approx(drive.values - DRIVE_LO)
    assert near[1][0].values == pytest.approx(probe.values - PROBE_LO)
    assert near[0][0].values == pytest.approx(amplitude.values)
    assert amplitudes[readout.id] is near[0][0]
    # the swept parameters are the oscillators of the calibration
    assert calib["q0/drive"].oscillator.frequency is real[0][0]
    assert calib["q0/probe"].oscillator.frequency is near[1][0]
    # the acquisition demodulates with the swept probe oscillator
    assert calib["q0/acquisition"].oscillator is calib["q0/probe"].oscillator


def _nested(channels, configs, values):  # noqa: F811
    """Split a real-time drive amplitude sweep followed by a drive frequency one."""
    calib = calibration(["q0/drive"], channels, INSTRUMENTS, configs)
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence([("q0/drive", pulse)])
    amplitude = Sweeper(
        parameter=Parameter.amplitude, values=np.array([0.1, 0.5]), pulses=[pulse]
    )
    frequency = Sweeper(
        parameter=Parameter.frequency, values=values, channels=["q0/drive"]
    )
    return split([[amplitude], [frequency]], [sequence], channels, configs, calib, {})


def test_split_uneven_frequency_nested(channels, configs):  # noqa: F811
    """An uneven frequency sweep nested in a real-time sweep goes to near time.

    LabOne Q fails to compile it in real time (``c_freq_osc_0`` already declared).
    """
    values = 6.3e9 + np.array([-1e6, 0, 2e6])
    near, real, axes = _nested(channels, configs, values)

    assert len(near) == 1 and len(real) == 1
    assert axes == [1, None, 0]


@pytest.mark.parametrize(
    "values",
    [np.linspace(6.29e9, 6.31e9, 5), 6.3e9 + np.array([0.0, 1e6])],
    ids=["linspace", "two-points"],
)
def test_split_even_frequency_nested(channels, configs, values):  # noqa: F811
    """Evenly spaced grids (and two-point ones) stay in real time."""
    near, real, axes = _nested(channels, configs, values)

    assert len(near) == 0 and len(real) == 2
    assert axes == [None, 0, 1]


def test_split_uneven_frequency_alone(channels, configs):  # noqa: F811
    """Without an outer real-time sweep an uneven frequency sweep stays in real time."""
    calib = calibration(["q0/drive"], channels, INSTRUMENTS, configs)
    frequency = Sweeper(
        parameter=Parameter.frequency,
        values=6.3e9 + np.array([-1e6, 0, 2e6]),
        channels=["q0/drive"],
    )
    near, real, axes = split([[frequency]], [], channels, configs, calib, {})

    assert len(near) == 0 and len(real) == 1
    assert axes == [None, 0]


def test_split_drive_amplitude(channels, configs):  # noqa: F811
    """The amplitude of a drive pulse is swept in real time, a probe one in near."""
    calib = calibration(IDS, channels, INSTRUMENTS, configs)
    seq, pulse, readout = sequence()
    values = np.array([0.1, 0.5, 1.0])
    sweepers = [
        [Sweeper(parameter=Parameter.amplitude, values=values, pulses=[readout])],
        [Sweeper(parameter=Parameter.amplitude, values=values, pulses=[pulse])],
    ]
    amplitudes = {}

    near, real, axes = split(sweepers, [seq], channels, configs, calib, amplitudes)

    assert axes == [0, None, 1]
    assert amplitudes[readout.id] is near[0][0]
    assert amplitudes[pulse.id] is real[0][0]
