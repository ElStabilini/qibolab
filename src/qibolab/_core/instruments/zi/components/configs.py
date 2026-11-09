"""Configuration kinds for Zurich Instruments channels.

Register them in a platform with ``ConfigKinds.extend([ZiConfigs])``.

The output and input ranges are per channel. Channels on the same physical port (e.g.
a drive channel and its ef drive, or all the probes of one QA channel) must hold the
same ``power_range``; LabOne Q reports a mismatch.
"""

from typing import Literal

from qibolab._core.components import AcquisitionConfig, DcConfig, IqConfig

__all__ = ["ZiAcquisitionConfig", "ZiConfigs", "ZiDcConfig", "ZiIqConfig"]


class ZiDcConfig(DcConfig):
    """DC channel config for an HDAWG output."""

    kind: Literal["zi-dc"] = "zi-dc"

    power_range: float
    """Output range in V."""


class ZiIqConfig(IqConfig):
    """IQ channel config for an SHFSG or SHFQA output."""

    kind: Literal["zi-iq"] = "zi-iq"

    # TODO: double check the values, apparently there's confusion between dB and dBm
    power_range: float
    """Output range in dBm.

    Possible values are -30 to 10, in steps of 5 dB. At 10 dBm the full scale is
    1 V peak.
    """


class ZiAcquisitionConfig(AcquisitionConfig):
    """Acquisition channel config for an SHFQA input."""

    kind: Literal["zi-acquisition"] = "zi-acquisition"

    # TODO: double check the values, apparently there's confusion between dB and dBm
    power_range: float
    """Input range in dBm.

    Possible values are -50 to 10, in steps of 5 dB.
    """


ZiConfigs = ZiIqConfig | ZiDcConfig | ZiAcquisitionConfig
