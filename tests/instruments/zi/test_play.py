"""Tests emulated execution, replicating the calls of qibocal protocols."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.pulses import Pulse, Readout, Rectangular
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
    sweepers = [
        [
            Sweeper(
                parameter=Parameter.frequency,
                values=configs[f"q{q}/probe"].frequency + np.linspace(-2e6, 2e6, 5),
                channels=[f"q{q}/probe"],
            )
            for q in range(NQUBITS)
        ]
    ]
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
    frequencies = [
        Sweeper(
            parameter=Parameter.frequency,
            values=configs[probe].frequency + np.linspace(-2e6, 2e6, 5),
            channels=[probe],
        )
        for probe in probes
    ]
    sweepers = [[amplitude], frequencies]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)


def test_unsupported(connected, configs):  # noqa: F811
    sequence = readouts([0])
    with pytest.raises(NotImplementedError):
        connected.play(configs, [sequence], ExecutionParameters(nshots=1), [])
    opts = options(AveragingMode.CYCLIC)
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence([("q0/drive", drive)]) | sequence
    for sweeper in (
        Sweeper(
            parameter=Parameter.frequency,
            values=np.array([6.3e9]),
            channels=["q0/drive"],
        ),
        Sweeper(parameter=Parameter.duration, values=np.array([40]), pulses=[drive]),
    ):
        with pytest.raises(NotImplementedError):
            connected.play(configs, [sequence], opts, [[sweeper]])
    for update in (
        {"fast_reset": True},
        {"averaging_mode": AveragingMode.SEQUENTIAL},
        {"acquisition_type": AcquisitionType.DISCRIMINATION},
        {"acquisition_type": AcquisitionType.RAW},
    ):
        with pytest.raises(NotImplementedError):
            connected.play(configs, [sequence], opts.model_copy(update=update), [])
