"""Tests ``components/configs.py``."""

import pytest

pytest.importorskip("laboneq")

from qibolab._core.components.filters import ExponentialFilter
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiConfigs,
    ZiDcConfig,
    ZiIqConfig,
)
from qibolab._core.parameters import ConfigKinds, Parameters


@pytest.fixture(autouse=True)
def zi_kinds():
    ConfigKinds.extend([ZiConfigs])
    yield
    ConfigKinds.reset()


def test_parameters_roundtrip():
    configs = {
        "q0/drive": ZiIqConfig(frequency=6.3e9, power_range=10),
        "q0/probe": ZiIqConfig(frequency=7.1e9, power_range=10),
        "q0/flux": ZiDcConfig(
            offset=0.1,
            power_range=5,
            filters=[ExponentialFilter(amplitude=-0.01, tau=100.0)],
        ),
        "q0/acquisition": ZiAcquisitionConfig(
            delay=200, smearing=0, power_range=-30, threshold=0.5, iq_angle=1.2
        ),
    }
    parameters = Parameters(configs=configs)

    reloaded = Parameters.model_validate_json(parameters.model_dump_json())

    assert reloaded == parameters
    for name, config in configs.items():
        assert type(reloaded.configs[name]) is type(config)
