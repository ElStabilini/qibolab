"""Tests emulated execution, replicating the calls of qibocal protocols."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.simulator.output_simulator import OutputSimulator

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.zi.pulse import NS
from qibolab._core.pulses import Delay, Gaussian, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import NQUBITS
from .test_calibration import channels, configs  # noqa: F401

NSHOTS = 7


@pytest.fixture
def connected(controller, channels):  # noqa: F811
    controller.channels = channels
    controller.connect()
    yield controller
    controller.disconnect()


def readouts(qubits) -> PulseSequence:
    """Multiplexed readout, as the ``MZ`` natives of a platform."""
    probe = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    return PulseSequence(
        [(f"q{q}/acquisition", Readout.from_probe(probe.new())) for q in qubits]
    )


def drives(qubits, duration: int = 200) -> PulseSequence:
    """Drive pulse on each qubit, then a delayed readout, as qubit spectroscopy."""
    drive = Pulse(duration=duration, amplitude=1.0, envelope=Rectangular())
    sequence = PulseSequence()
    for q in qubits:
        sequence.append((f"q{q}/drive", drive.new()))
        sequence.append((f"q{q}/acquisition", Delay(duration=duration)))
    return sequence | readouts(qubits)


def rabi(qubits, rx90: bool = False, duration: int = 48) -> PulseSequence:
    """Drive pulse then delayed readout, as the ``rabi_amplitude_signal`` sequence.

    With ``rx90``, the same pulse is played twice, as qibocal does to calibrate a
    pi/2 pulse, so the same swept amplitude is used by two operations.
    """
    drive = Pulse(duration=duration, amplitude=1.0, envelope=Gaussian(rel_sigma=1 / 6))
    sequence = PulseSequence()
    for q in qubits:
        pulse = drive.new()
        for _ in range(2 if rx90 else 1):
            sequence.append((f"q{q}/drive", pulse))
        sequence.append((f"q{q}/acquisition", Delay(duration=duration)))
    return sequence | readouts(qubits)


def states(qubits) -> list[PulseSequence]:
    """Readout of the ground and of the excited state, as ``single_shot_classification``.

    The second sequence is the ``RX`` natives aligned before the ``MZ`` ones, so it
    carries the ``Align`` that ``|`` inserts.
    """
    drive = Pulse(duration=48, amplitude=1.0, envelope=Gaussian(rel_sigma=1 / 6))
    excited = PulseSequence([(f"q{q}/drive", drive.new()) for q in qubits])
    return [readouts(qubits), excited | readouts(qubits)]


def drive_pulses(sequence: PulseSequence) -> list[Pulse]:
    """The distinct pulses played on the drive channels, as qibocal sweeps them."""
    return list(
        {
            p.id: p
            for id, p in sequence
            if id.endswith("/drive") and isinstance(p, Pulse)
        }.values()
    )


def frequencies(configs, channels, points: int = 5) -> list[Sweeper]:  # noqa: F811
    """Absolute frequency sweeper of each channel, as the qibocal protocols."""
    return [
        Sweeper(
            parameter=Parameter.frequency,
            values=configs[channel].frequency + np.linspace(-2e6, 2e6, points),
            channels=[channel],
        )
        for channel in channels
    ]


def options(averaging_mode) -> ExecutionParameters:
    return ExecutionParameters(
        nshots=NSHOTS,
        relaxation_time=1000,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=averaging_mode,
    )


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_resonator_spectroscopy(connected, configs, averaging_mode):  # noqa: F811
    sequence = readouts(range(NQUBITS))
    probes = [f"q{q}/probe" for q in range(NQUBITS)]
    sweepers = [frequencies(configs, probes)]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_multiple_sequences(connected, configs, averaging_mode):  # noqa: F811
    sequences = [readouts([0, 1]), readouts([0])]
    opts = options(averaging_mode)

    results = connected.play(configs, sequences, opts, [])

    assert len(results) == 3
    for result in results.values():
        assert result.shape == opts.results_shape([])


def test_resonator_punchout(connected, configs):  # noqa: F811
    sequence = readouts(range(NQUBITS))
    probes = [f"q{q}/probe" for q in range(NQUBITS)]
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    sweepers = [[amplitude], frequencies(configs, probes)]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_qubit_spectroscopy(connected, configs, averaging_mode):  # noqa: F811
    """The drive frequency is swept in real time, inside the acquisition loop."""
    sequence = drives(range(NQUBITS))
    drive = [f"q{q}/drive" for q in range(NQUBITS)]
    sweepers = [frequencies(configs, drive)]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize("rx90", [False, True])
def test_rabi_amplitude_signal(connected, configs, rx90):  # noqa: F811
    """The amplitude of the drive pulses is swept in real time."""
    sequence = rabi(range(NQUBITS), rx90=rx90)
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.0, 1.0, 0.25),
        pulses=drive_pulses(sequence),
    )
    sweepers = [[amplitude]]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_rabi_amplitude_with_probe_sweep(connected, configs):  # noqa: F811
    """A real-time drive amplitude nested in a near-time probe one keeps its order."""
    sequence = rabi([0])
    drive = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.0, 1.0, 0.25),
        pulses=drive_pulses(sequence),
    )
    probe = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 0.4, 0.1),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    sweepers = [[drive], [probe]]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_near_and_real_time_axes(connected, configs, averaging_mode):  # noqa: F811
    """A near-time and a real-time sweep come back in the order qibolab asked.

    The two have different lengths, so a swapped axis changes the shape. The
    real-time drive sweep is given first, although LabOne Q returns it last.
    """
    sequence = drives([0])
    sweepers = [
        frequencies(configs, ["q0/drive"], points=3),
        frequencies(configs, ["q0/probe"], points=5),
    ]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_single_shot_classification(connected, configs):  # noqa: F811
    """Two sequences, no sweepers, one result per shot."""
    sequences = states(range(NQUBITS))
    opts = options(AveragingMode.SINGLESHOT)

    results = connected.play(configs, sequences, opts, [])

    acquisitions = [ro for sequence in sequences for _, ro in sequence.acquisitions]
    assert set(results) == {ro.id for ro in acquisitions}
    assert len(results) == 2 * NQUBITS
    for result in results.values():
        assert result.shape == (NSHOTS, 2)
        assert not np.isnan(result).any()


def test_dispersive_shift(connected, configs):  # noqa: F811
    """The two states of the classification, with one probe-frequency sweep."""
    sequences = states(range(NQUBITS))
    probes = [f"q{q}/probe" for q in range(NQUBITS)]
    sweepers = [frequencies(configs, probes)]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, sequences, opts, sweepers)

    assert len(results) == 2 * NQUBITS
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


RX = 48
"""Duration of the ``RX`` pulse of the duration sweeps, in ns."""

WAITS = np.arange(16, 56, 8)
"""Swept durations, on the 8 ns grid, which a swept readout start must be on."""


def rx() -> Pulse:
    return Pulse(duration=RX, amplitude=0.5, envelope=Gaussian(rel_sigma=1 / 6))


def readout() -> Readout:
    return Readout.from_probe(
        Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    )


def duration(pulses, values=WAITS) -> Sweeper:
    return Sweeper(parameter=Parameter.duration, values=values, pulses=pulses)


def rabi_length(qubits) -> tuple[PulseSequence, list[Pulse | Delay]]:
    """Drive pulse, readout delayed by a ``Delay``, both swept, as ``rabi_length``."""
    sequence = PulseSequence()
    swept = []
    for q in qubits:
        pulse, delay = rx(), Delay(duration=16)
        sequence.append((f"q{q}/drive", pulse))
        sequence.append((f"q{q}/acquisition", delay))
        sequence.append((f"q{q}/acquisition", readout()))
        swept += [pulse, delay]
    return sequence, swept


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_rabi_length_signal(connected, configs, averaging_mode):  # noqa: F811
    sequence, swept = rabi_length(range(NQUBITS))
    sweepers = [[duration(swept)]]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert len(results) == NQUBITS
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_rabi_length_chunked(connected, configs, averaging_mode):  # noqa: F811
    """A length sweep too big for the waveform memory runs in chunks."""
    sequence, swept = rabi_length(range(NQUBITS))
    sweepers = [[duration(swept, np.linspace(16, 1000, 100) // 8 * 8)]]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_t1_signal(connected, configs, averaging_mode):  # noqa: F811
    """One ``Delay`` object before the readout of every qubit, as ``t1_signal``."""
    delay = Delay(duration=0)
    sequence = PulseSequence()
    for q in range(NQUBITS):
        sequence.append((f"q{q}/drive", rx()))
        sequence.append((f"q{q}/acquisition", Delay(duration=RX)))
        sequence.append((f"q{q}/acquisition", delay))
        sequence.append((f"q{q}/acquisition", readout()))
    sweepers = [[duration(NQUBITS * [delay], np.arange(0, 2000, 400))]]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_ramsey_signal(connected, configs):  # noqa: F811
    """The same pulse twice around a swept ``Delay``, as ``ramsey_signal``."""
    sequence = PulseSequence()
    delays = []
    for q in range(NQUBITS):
        pulse, wait, ro_wait = rx(), Delay(duration=0), Delay(duration=0)
        sequence.extend([(f"q{q}/drive", pulse), (f"q{q}/drive", wait)])
        sequence.append((f"q{q}/drive", pulse))
        sequence.append((f"q{q}/acquisition", Delay(duration=2 * RX)))
        sequence.append((f"q{q}/acquisition", ro_wait))
        sequence.append((f"q{q}/acquisition", readout()))
        delays += [wait, ro_wait]
    sweepers = [[duration(delays)]]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_spin_echo_signal(connected, configs):  # noqa: F811
    """The same ``Delay`` object repeated on each channel, as ``spin_echo_signal``."""
    n = 2
    sequence = PulseSequence()
    delays = []
    for q in range(NQUBITS):
        drive, acquisition = f"q{q}/drive", f"q{q}/acquisition"
        rx90, pi = rx(), rx()
        wait, ro_wait = Delay(duration=0), Delay(duration=0)
        sequence.append((drive, rx90))
        for _ in range(n):
            sequence.extend([(drive, wait), (acquisition, ro_wait), (drive, pi)])
            sequence.extend([(drive, wait), (acquisition, ro_wait)])
        sequence.append((drive, rx90))
        sequence.append((acquisition, Delay(duration=(2 + n) * RX)))
        sequence.append((acquisition, readout()))
        delays += 2 * n * [wait] + 2 * n * [ro_wait]
    sweepers = [[duration(delays)]]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_duration_with_probe_sweep(connected, configs, averaging_mode):  # noqa: F811
    """A real-time duration given before a near-time probe frequency.

    The two sweeps have different sizes, so a wrong axis order changes the shape.
    """
    sequence, swept = rabi_length([0])
    sweepers = [[duration(swept, WAITS[:4])], frequencies(configs, ["q0/probe"], 3)]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_rabi_length_align(connected, configs):  # noqa: F811
    """A swept pulse aligned with the readout, as ``rabi_length`` with ``use_align``.

    The lengths are off the 8 ns grid: the readout section starts on the grid.
    """
    pulse = rx()
    sequence = PulseSequence([("q0/drive", pulse)])
    sequence.align(["q0/drive", "q0/acquisition"])
    sequence.append(("q0/acquisition", readout()))
    sweepers = [[duration([pulse], np.array([20, 36, 52]))]]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def start(compiled, controller, signal: str) -> float:
    """Time in ns of the first output sample of ``signal``, in the first shot."""
    channel = controller.session.device_setup.logical_signal_by_uid(signal)
    snippet = OutputSimulator(compiled).get_snippet(
        channel.physical_channel, start=0, output_length=2e-6
    )
    return snippet.time[np.flatnonzero(np.abs(snippet.wave) > 1e-6)[0]] / NS


@pytest.mark.parametrize(
    "length,expected", [(48, 48), (50, 56)], ids=["on-grid", "off-grid"]
)
def test_align(connected, configs, monkeypatch, length, expected):  # noqa: F811
    """A readout aligned to one drive only starts when that drive ends.

    The other drive is longer and not aligned, so it does not delay the readout.
    After a pulse off the 8 ns grid, the readout starts at the next grid point.
    """
    sequence = PulseSequence(
        [
            ("q0/drive", Pulse(duration=length, amplitude=0.5, envelope=Rectangular())),
            ("q1/drive", Pulse(duration=200, amplitude=0.5, envelope=Rectangular())),
        ]
    )
    sequence.align(["q0/drive", "q0/acquisition"])
    sequence.append(("q0/acquisition", readout()))
    compiled = []
    compile = connected.session.compile
    monkeypatch.setattr(
        connected.session,
        "compile",
        lambda experiment: compiled.append(compile(experiment)) or compiled[-1],
    )

    connected.play(configs, [sequence], options(AveragingMode.CYCLIC), [])

    drive = start(compiled[0], connected, "q0/drive")
    probe = start(compiled[0], connected, "q0/probe")
    assert probe - drive == pytest.approx(expected)


def test_unsupported(connected, configs):  # noqa: F811
    sequence = readouts([0])
    opts = options(AveragingMode.CYCLIC)
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence([("q0/drive", drive)]) | sequence
    for parameter, pulse in [
        (Parameter.duration_interpolated, drive),
        (Parameter.duration, sequence.acquisitions[0][1]),
    ]:
        duration = Sweeper(parameter=parameter, values=np.array([40]), pulses=[pulse])
        with pytest.raises(NotImplementedError, match="duration"):
            connected.play(configs, [sequence], opts, [[duration]])
    # the drive frequency is swept in real time, the probe one in near time
    parallel = frequencies(configs, ["q0/drive", "q0/probe"])
    with pytest.raises(NotImplementedError, match="real time"):
        connected.play(configs, [sequence], opts, [parallel])
    for update in (
        {"fast_reset": True},
        {"averaging_mode": AveragingMode.SEQUENTIAL},
        {"acquisition_type": AcquisitionType.RAW},
    ):
        with pytest.raises(NotImplementedError):
            connected.play(configs, [sequence], opts.model_copy(update=update), [])
