"""Tests flux offsets and flux pulses, replicating the calls of qibocal protocols."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.execution_parameters import AcquisitionType, AveragingMode
from qibolab._core.instruments.zi.controller import calibration
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.pulses import Delay, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import INSTRUMENTS, NQUBITS
from .test_calibration import channels, configs  # noqa: F401
from .test_play import (  # noqa: F401
    RX,
    connected,
    drives,
    duration,
    frequencies,
    options,
    readout,
    readouts,
    rx,
    start,
)

QUBITS = range(NQUBITS)


def offsets(configs, channels, points: int = 3) -> list[Sweeper]:  # noqa: F811
    """Absolute offset sweeper of each channel, centred on its configured offset."""
    return [
        Sweeper(
            parameter=Parameter.offset,
            values=configs[channel].offset + np.linspace(-0.1, 0.1, points),
            channels=[channel],
        )
        for channel in channels
    ]


def fluxes(qubits) -> list[str]:
    return [f"q{q}/flux" for q in qubits]


def test_split_offset(channels, configs):  # noqa: F811
    """An offset sweep replaces the voltage offset, unchanged, in near time."""
    ids = ["q0/flux", "q0/drive"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    (bias,) = offsets(configs, ["q0/flux"])
    (drive,) = frequencies(configs, ["q0/drive"])

    near, real, axes = split(
        [[bias], [drive]], [PulseSequence()], channels, configs, calib, {}
    )

    assert axes == [0, None, 1]
    assert calib["q0/flux"].voltage_offset is near[0][0]
    assert near[0][0].values == pytest.approx(bias.values)
    assert real[0][0] is calib["q0/drive"].oscillator.frequency


def _check(results, sequences, shape):
    acquisitions = {ro.id for seq in sequences for _, ro in seq.acquisitions}
    assert set(results) == acquisitions
    for result in results.values():
        assert result.shape == shape
        assert not np.isnan(result).any()


def test_resonator_flux_dependence(connected, configs):  # noqa: F811
    """Near-time offsets around near-time probe frequencies."""
    sequence = readouts(QUBITS)
    sweepers = [
        offsets(configs, fluxes(QUBITS)),
        frequencies(configs, [f"q{q}/probe" for q in QUBITS]),
    ]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_qubit_flux_dependence(connected, configs, averaging_mode):  # noqa: F811
    """Near-time offsets around real-time drive frequencies keep their order."""
    sequence = drives(QUBITS)
    sweepers = [
        offsets(configs, fluxes(QUBITS)),
        frequencies(configs, [f"q{q}/drive" for q in QUBITS]),
    ]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


def test_qubit_crosstalk(connected, configs):  # noqa: F811
    """The offset of another qubit's flux line, around the drive frequencies."""
    sequence = drives([0, 2])
    sweepers = [
        offsets(configs, ["q1/flux"]),
        frequencies(configs, ["q0/drive", "q2/drive"]),
    ]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


@pytest.fixture
def runs(connected, monkeypatch):  # noqa: F811
    """Compiled experiments run by the controller."""
    compiled = []
    run = connected.session.run

    def spy(experiment, *args, **kwargs):
        compiled.append(experiment)
        return run(experiment, *args, **kwargs)

    monkeypatch.setattr(connected.session, "run", spy)
    return compiled


def test_offsets_restored(connected, configs, runs):  # noqa: F811
    """After an offset sweep, the swept lines go back to their configured offsets."""
    swept = fluxes([1, 3])
    sweepers = [offsets(configs, swept)]

    connected.play(configs, [readouts([0])], options(AveragingMode.CYCLIC), sweepers)

    assert len(runs) == 2
    (hdawg,) = [
        init
        for init in runs[1].scheduled_experiment.artifacts.initializations
        if init.device_uid == "hdawg"
    ]
    assert {o.channel: o.offset for o in hdawg.outputs} == {
        connected.channels[id].port: configs[id].offset for id in swept
    }


def test_offsets_not_restored_without_sweep(connected, configs, runs):  # noqa: F811
    sweepers = [frequencies(configs, ["q0/probe"])]

    connected.play(configs, [readouts([0])], options(AveragingMode.CYCLIC), sweepers)

    assert len(runs) == 1


def amplitude(pulses) -> Sweeper:
    values = np.linspace(0.1, 0.5, 3)
    return Sweeper(parameter=Parameter.amplitude, values=values, pulses=pulses)


def flux_pulse(length: int = 40) -> Pulse:
    return Pulse(duration=length, amplitude=0.3, envelope=Rectangular())


def discrimination(averaging_mode):
    return options(averaging_mode).model_copy(
        update={"acquisition_type": AcquisitionType.DISCRIMINATION}
    )


def test_split_flux_pulse(channels, configs):  # noqa: F811
    """Amplitude and duration of a flux pulse are swept in real time."""
    ids = ["q0/flux"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    pulse = flux_pulse()
    seq = PulseSequence([("q0/flux", pulse)])
    swept = {}

    near, real, axes = split(
        [[duration([pulse])], [amplitude([pulse])]],
        [seq],
        channels,
        configs,
        calib,
        swept,
    )

    assert axes == [None, 0, 1] and len(near) == 0
    assert real == [[swept[pulse.id]["length"]], [swept[pulse.id]["amplitude"]]]


def chevron(low: int, high: int, dt: int = 8):
    """The ``chevron_sequence`` of qibocal, for the ``CZ`` native of a pair.

    One ``Delay`` object, swept with the flux pulse, waits on the drive of the high
    qubit and on both acquisition channels.
    """
    sequence = PulseSequence([(f"q{low}/drive", rx()), (f"q{high}/drive", rx())])
    flux, pulse = f"q{high}/flux", flux_pulse(WAITS_MAX)
    sequence += [(flux, Delay(duration=RX)), (flux, pulse)]
    wait, dt_delay = Delay(duration=WAITS_MAX), Delay(duration=dt)
    for q in (low, high):
        acquisition = f"q{q}/acquisition"
        sequence += [(acquisition, Delay(duration=RX)), (acquisition, wait)]
        sequence += [(acquisition, dt_delay), (acquisition, Delay(duration=RX))]
    drive = f"q{high}/drive"
    sequence += [(drive, wait), (drive, dt_delay), (drive, rx())]
    sequence += [(f"q{q}/acquisition", readout()) for q in (low, high)]
    return sequence, pulse, wait


WAITS_MAX = 80
"""Longest flux pulse of the chevron, in ns."""


@pytest.mark.parametrize(
    "acquisition_type", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
def test_chevron(connected, configs, acquisition_type):  # noqa: F811
    """Duration of the flux pulse and of the delays, then its amplitude, as
    ``chevron_signal`` (integration) and ``chevron`` (discrimination)."""
    sequence, pulse, wait = chevron(0, 1)
    sweepers = [
        [duration([pulse, wait], np.arange(16, WAITS_MAX, 16))],
        [amplitude([pulse])],
    ]
    opts = options(AveragingMode.CYCLIC).model_copy(
        update={"acquisition_type": acquisition_type}
    )

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


def test_t1_flux(connected, configs):  # noqa: F811
    """Flux amplitude, then the wait, on a flux pulse starting at 0 ns, as ``t1_flux``.

    The wait is a ``Delay`` before the readout and the flux pulse during it.
    """
    sequence = PulseSequence()
    delays, pulses = [], []
    for q in (0, 1):
        delay, pulse = Delay(duration=0), flux_pulse(0)
        sequence += [(f"q{q}/drive", rx()), (f"q{q}/acquisition", Delay(duration=RX))]
        sequence += [(f"q{q}/flux", Delay(duration=RX)), (f"q{q}/flux", pulse)]
        sequence += [(f"q{q}/acquisition", delay), (f"q{q}/acquisition", readout())]
        delays.append(delay)
        pulses.append(pulse)
    sweepers = [[amplitude(pulses)], [duration(delays + pulses, np.arange(0, 80, 16))]]
    opts = discrimination(AveragingMode.SINGLESHOT)

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


def test_flux_gate(connected, configs):  # noqa: F811
    """A flux pulse between two pi/2 pulses, its duration swept, as ``flux_gate``."""
    sequence = PulseSequence()
    swept = []
    for q in QUBITS:
        drive, acquisition, flux = f"q{q}/drive", f"q{q}/acquisition", f"q{q}/flux"
        rx90, pulse = rx(), flux_pulse(WAITS_MAX)
        wait, ro_wait = Delay(duration=WAITS_MAX), Delay(duration=WAITS_MAX)
        sequence += [(drive, rx90), (flux, Delay(duration=RX)), (flux, pulse)]
        sequence += [(drive, wait), (drive, rx90), (acquisition, ro_wait)]
        sequence += [(acquisition, Delay(duration=2 * RX)), (acquisition, readout())]
        swept += [pulse, wait, ro_wait]
    sweepers = [[duration(swept, np.arange(16, WAITS_MAX, 16))]]
    opts = discrimination(AveragingMode.SINGLESHOT)

    results = connected.play(configs, [sequence], opts, sweepers)

    _check(results, [sequence], opts.results_shape(sweepers))


def test_flux_pulse_output(connected, configs, monkeypatch):  # noqa: F811
    """A flux pulse plays after the drive, on top of the offset of its line."""
    sequence = PulseSequence([("q0/drive", rx())])
    sequence += [("q0/flux", Delay(duration=RX)), ("q0/flux", flux_pulse(64))]
    compiled = []
    compile = connected.session.compile
    monkeypatch.setattr(
        connected.session,
        "compile",
        lambda experiment: compiled.append(compile(experiment)) or compiled[-1],
    )
    configs["q0/flux"] = configs["q0/flux"].model_copy(update={"offset": 0.0})

    connected.play(configs, [sequence], options(AveragingMode.CYCLIC), [])

    flux = start(compiled[0], connected, "q0/flux")
    drive = start(compiled[0], connected, "q0/drive")
    assert flux - drive == pytest.approx(RX)
