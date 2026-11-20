"""Tests the LabOne Q pulses built from the qibolab envelopes."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.core.utilities.pulse_sampler import sample_pulse

from qibolab._core.instruments.zi.pulse import NS, kernel, waveform
from qibolab._core.pulses import (
    Custom,
    Drag,
    Gaussian,
    GaussianSquare,
    Pulse,
    Rectangular,
    Snz,
)

SAMPLING_RATE = 2.0
DURATION = 48
SAMPLES = int(DURATION * SAMPLING_RATE)

ENVELOPES = [
    Rectangular(),
    Gaussian(rel_sigma=1 / 6),
    Drag(rel_sigma=1 / 6, beta=0.7),
    GaussianSquare(risefall=10, sigma=4),
    Snz(t_idling=10, b_amplitude=0.3),
    Custom(i_=np.linspace(0, 1, SAMPLES), q_=np.linspace(1, 0, SAMPLES)),
]


def sample(envelope, subslice=None) -> np.ndarray:
    """Samples LabOne Q generates for the pulse of ``envelope``."""
    pulse = waveform(envelope, DURATION)
    return sample_pulse(
        signal_type="iq",
        sampling_rate=SAMPLING_RATE / NS,
        length=pulse.length,
        amplitude=1.0,
        pulse_function=pulse.function,
        pulse_parameters=pulse.pulse_parameters,
        subslice=subslice,
    )


@pytest.mark.parametrize("envelope", ENVELOPES, ids=lambda e: e.kind)
def test_samples(envelope):
    """The functional pulse plays exactly the samples of the qibolab pulse."""
    pulse = Pulse(duration=DURATION, amplitude=1.0, envelope=envelope)
    expected = pulse.i(SAMPLING_RATE) + 1j * pulse.q(SAMPLING_RATE)
    np.testing.assert_allclose(sample(envelope), expected)
    # LabOne Q may sample a part of the pulse only, when merging waveforms
    np.testing.assert_allclose(sample(envelope, (10, 40)), expected[10:40])


def test_same_object():
    """Equal envelopes give one pulse object, as an SHFQA output holds one waveform."""
    assert waveform(Gaussian(rel_sigma=0.2), 40) is waveform(
        Gaussian(rel_sigma=0.2), 40
    )
    assert waveform(Gaussian(rel_sigma=0.2), 40) is not waveform(Rectangular(), 40)


@pytest.mark.parametrize("angle", [0.0, 0.4, -2.0])
def test_kernel_rotation(angle):
    """Integrating with the kernel rotates as qibolab's classification.

    The SHFQA integrates the signal times the conjugated weights; the real part of
    the result is the first component of the I/Q point rotated by ``angle``, which
    the threshold is compared to.
    """
    weights = kernel(DURATION, angle)
    assert weights.length == pytest.approx(DURATION * NS)
    signal = 0.3 - 0.8j
    rotated = signal * np.conj(weights.amplitude)
    c, s = np.cos(angle), np.sin(angle)
    assert rotated.real == pytest.approx(c * signal.real - s * signal.imag)
    assert kernel(DURATION, angle) is weights
