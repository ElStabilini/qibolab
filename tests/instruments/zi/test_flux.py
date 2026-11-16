"""Tests flux offset sweeps, replicating the calls of the qibocal flux protocols."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.zi.controller import calibration
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import INSTRUMENTS, NQUBITS
from .test_calibration import channels, configs  # noqa: F401
from .test_play import connected, drives, frequencies, options, readouts  # noqa: F401

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
