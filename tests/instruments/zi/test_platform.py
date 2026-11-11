"""Tests ``platform.py``."""

import pytest

pytest.importorskip("laboneq")

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.zi import ZiInstrument, infer_los, map_ports
from qibolab._core.instruments.zi.controller import _PORTS
from qibolab._core.instruments.zi.platform import _CHANNELS
from qibolab._core.qubits import Qubit

from .conftest import CLUSTER, INSTRUMENTS, NQUBITS


def _cluster(ports):
    """Cluster with the conftest instruments and the given ports."""
    return {name: (INSTRUMENTS[name], p) for name, p in ports.items()}


def test_map_ports(channels, qubits):
    assert len(channels) == 4 * NQUBITS
    for i, qubit in enumerate(qubits.values()):
        assert channels[qubit.drive] == IqChannel(device="shfsg", path=str(i))
        assert channels[qubit.flux] == DcChannel(device="hdawg", path=str(i))
        assert channels[qubit.probe] == IqChannel(device="shfqa", path="0")
        assert channels[qubit.acquisition] == AcquisitionChannel(
            device="shfqa", path="0", probe=qubit.probe
        )


def test_map_ports_couplers(qubits):
    couplers = {"c0": Qubit.coupler("c0")}
    ports = {"hdawg": {0: ["q0"], 1: ["c0"]}}
    channels = map_ports(_cluster(ports), qubits, couplers)
    assert channels == {
        "q0/flux": DcChannel(device="hdawg", path="0"),
        "coupler_c0/flux": DcChannel(device="hdawg", path="1"),
    }


def test_map_ports_drive_extra(qubits):
    qubits["q1"].drive_extra[(1, 2)] = "q1/drive12"
    channels = map_ports(_cluster({"shfsg": {3: ["q1"]}}), qubits)
    assert channels == {
        "q1/drive": IqChannel(device="shfsg", path="3"),
        "q1/drive12": IqChannel(device="shfsg", path="3"),
    }


def test_map_ports_same_element_twice_on_one_port(qubits):
    channels = map_ports(_cluster({"shfsg": {0: ["q0", "q0"]}}), qubits)
    assert channels == {"q0/drive": IqChannel(device="shfsg", path="0")}


def test_map_ports_shared_probe():
    qubits = {
        f"q{i}": Qubit(probe="feedline/probe", acquisition=f"q{i}/acquisition")
        for i in range(2)
    }
    channels = map_ports(_cluster({"shfqa": {0: ["q0", "q1"]}}), qubits)
    assert channels["feedline/probe"] == IqChannel(device="shfqa", path="0")


@pytest.mark.parametrize(
    "ports",
    [
        # one element on two ports
        {"shfsg": {0: ["q0"], 1: ["q0"]}},
        {"shfqa": {0: ["q0"], 1: ["q0"]}},
        # two elements resolving to the same channel id
        {"shfqa": {0: ["q0"], 1: ["q1"]}},
    ],
)
def test_map_ports_duplicate_channel(ports):
    qubits = {
        f"q{i}": Qubit(
            drive=f"q{i}/drive",
            probe="feedline/probe",
            acquisition=f"q{i}/acquisition",
        )
        for i in range(2)
    }
    with pytest.raises(ValueError):
        map_ports(_cluster(ports), qubits)


def test_map_ports_drive_extra_collision(qubits):
    qubits["q0"].drive_extra[(1, 2)] = qubits["q1"].drive
    with pytest.raises(ValueError):
        map_ports(_cluster({"shfsg": {0: ["q0"], 1: ["q1"]}}), qubits)


def test_map_ports_qubit_coupler_overlap(qubits):
    couplers = {"q0": Qubit.coupler("q0")}
    with pytest.raises(ValueError):
        map_ports(_cluster({"hdawg": {0: ["q0"]}}), qubits, couplers)


def test_map_ports_unknown_element(qubits):
    with pytest.raises(ValueError):
        map_ports(_cluster({"shfsg": {0: ["q99"]}}), qubits)


def test_map_ports_missing_channel(qubits):
    couplers = {"c0": Qubit.coupler("c0")}
    with pytest.raises(ValueError):
        map_ports(_cluster({"shfsg": {0: ["c0"]}}), qubits, couplers)


def test_map_ports_unsupported_kind(qubits):
    pqsc = ZiInstrument(kind="PQSC", serial="DEV10001")  # type: ignore
    with pytest.raises(ValueError):
        map_ports({"pqsc": (pqsc, {0: ["q0"]})}, qubits)


@pytest.mark.parametrize(
    "instrument,port",
    [
        (ZiInstrument(kind="SHFQA", serial="DEV0", options="SHFQA2"), 2),
        (ZiInstrument(kind="SHFSG", serial="DEV0", options="SHFSG4"), 4),
        (ZiInstrument(kind="HDAWG", serial="DEV0", options="HDAWG4/PC"), 4),
        (ZiInstrument(kind="SHFSG", serial="DEV0"), 8),
    ],
)
def test_map_ports_port_out_of_range(qubits, instrument, port):
    with pytest.raises(ValueError):
        map_ports({"dev": (instrument, {port: ["q0"]})}, qubits)


def test_channel_types_have_ports():
    for kind, types in _CHANNELS.items():
        for cls in types.values():
            assert (cls, kind) in _PORTS


def test_infer_los():
    los = infer_los(CLUSTER)
    assert len(los) == 2 * NQUBITS
    for i in range(NQUBITS):
        # SHFSG8: outputs share a synthesizer in pairs
        assert los[f"q{i}", False] == f"shfsg/synth{i // 2}/lo"
        assert los[f"q{i}", True] == "shfqa/o0/lo"


@pytest.mark.parametrize("options", ["SHFSG4", "SHFSG2", None])
def test_infer_los_one_synthesizer_per_port(options):
    sg = ZiInstrument(kind="SHFSG", serial="DEV0", options=options)
    los = infer_los({"sg": (sg, {0: ["q0"], 1: ["q1"]})})
    assert los == {("q0", False): "sg/o0/lo", ("q1", False): "sg/o1/lo"}
