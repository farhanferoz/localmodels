"""Which hardware a model can run on, checked before its server is started."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from localmodels.enums import DeviceType
from localmodels.hardware import inspect_hardware
from localmodels.serving.errors import ServingError, ServingErrorCode

if TYPE_CHECKING:
    from collections.abc import Collection

    from localmodels.hardware import HardwareReport

DEVICE_NAMES: Final = {
    DeviceType.CUDA: "a CUDA or ROCm GPU",
    DeviceType.MPS: "an Apple GPU",
    DeviceType.CPU: "only a CPU",
}


def require_device(supported: Collection[DeviceType], report: HardwareReport | None = None) -> DeviceType:
    """The measured device of this machine, when ``supported`` lists it.

    Raises:
        ServingError: ``not_implemented``, saying in plain words what the model needs and what this
            machine has, so a caller can show it before loading anything.
    """
    report = report or inspect_hardware()
    if report.device in supported:
        return report.device
    needs = " or ".join(DEVICE_NAMES[device] for device in DeviceType if device in supported)
    has = f"{DEVICE_NAMES[report.device]} ({report.device_name})"
    raise ServingError(ServingErrorCode.NOT_IMPLEMENTED, f"This model needs {needs}; this machine has {has}.")
