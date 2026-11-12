import pytest

pytest.importorskip("laboneq")

from qibolab._core.instruments.zi import ZiController, ZiInstrument, map_ports
from qibolab._core.qubits import Qubit

NQUBITS = 6

CLUSTER = {
    "hdawg": (
        ZiInstrument(kind="HDAWG", serial="DEV8001", options="HDAWG8/PC"),
        {i: [f"q{i}"] for i in range(NQUBITS)},
    ),
    "shfsg": (
        ZiInstrument(kind="SHFSG", serial="DEV12001", options="SHFSG8"),
        {i: [f"q{i}"] for i in range(NQUBITS)},
    ),
    "shfqa": (
        ZiInstrument(kind="SHFQA", serial="DEV12101", options="SHFQA4"),
        {0: [f"q{i}" for i in range(NQUBITS)]},
    ),
}
"""Fake rack, with invented serials."""

INSTRUMENTS = {name: instrument for name, (instrument, _) in CLUSTER.items()}

PQSC = "DEV10001"


@pytest.fixture
def qubits() -> dict[str, Qubit]:
    return {f"q{i}": Qubit.default(f"q{i}") for i in range(NQUBITS)}


@pytest.fixture
def channels(qubits):
    return map_ports(CLUSTER, qubits)


@pytest.fixture
def controller(channels) -> ZiController:
    return ZiController(
        address="127.0.0.1:8004",
        instruments=INSTRUMENTS,
        pqsc=PQSC,
        channels=channels,
        emulation=True,
    )
