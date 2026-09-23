"""Explicit test-condition compatibility between a requirement and a measurement.

Levels, best first:

``EXACT``          every condition the requirement states is reported identically
``COMPATIBLE``     every stated condition is reported and equivalent under the rules below
``UNCONDITIONED``  the requirement states no conditions; the measurement is at room or
                   unreported temperature (a general value)
``INCOMPATIBLE``   a stated condition is missing or different, or the measurement
                   was taken at a non-room temperature for an unconditioned requirement

Equivalence rules (documented, deliberately narrow):

* temperature: |ΔT| ≤ 1 K exact, ≤ 5 K compatible. A measurement marked
  ``temperature_regime=room`` is compatible with a stated 15–30 °C;
* environment: same :class:`EnvironmentClass` is compatible, and also the same
  ``medium`` text is exact. A different class is incompatible;
* exposure duration: within 1% exact, within 10% compatible, otherwise
  incompatible (24 h water uptake does not answer a 168 h question);
* relative humidity: within 1 %RH exact, within 5 %RH compatible.

A condition the requirement states but the measurement does not report can
never be established, so the measurement is ``INCOMPATIBLE`` for that
requirement (never silently assumed).
"""

from __future__ import annotations

from enum import IntEnum

from domains.uav_materials.schema import Quantity, TemperatureRegime, TestConditions

ROOM_RANGE_C = (15.0, 30.0)


class Compat(IntEnum):
    INCOMPATIBLE = 0
    UNCONDITIONED = 1
    COMPATIBLE = 2
    EXACT = 3


def _close(
    a: Quantity, b: Quantity, exact: float, compatible: float, relative: bool = False
) -> Compat:
    got = a.to(b.unit)
    diff = abs(got - b.value) / (abs(b.value) or 1.0) if relative else abs(got - b.value)
    return (
        Compat.EXACT
        if diff <= exact
        else Compat.COMPATIBLE
        if diff <= compatible
        else Compat.INCOMPATIBLE
    )


def _is_room(c: TestConditions) -> bool | None:
    if c.temperature_regime is TemperatureRegime.ROOM:
        return True
    if c.temperature is not None:
        t = c.temperature.to("degC")
        return ROOM_RANGE_C[0] <= t <= ROOM_RANGE_C[1]
    return None  # not reported


def compatibility(required: TestConditions, observed: TestConditions) -> tuple[Compat, str]:
    levels: list[Compat] = []
    stated = False
    if required.temperature is not None:
        stated = True
        if observed.temperature is not None:
            level = _close(observed.temperature, required.temperature, 1.0, 5.0)
        elif observed.temperature_regime is TemperatureRegime.ROOM:
            t = required.temperature.to("degC")
            level = (
                Compat.COMPATIBLE
                if ROOM_RANGE_C[0] <= t <= ROOM_RANGE_C[1]
                else Compat.INCOMPATIBLE
            )
        else:
            return Compat.INCOMPATIBLE, "test temperature not reported"
        if level is Compat.INCOMPATIBLE:
            return level, "different test temperature"
        levels.append(level)
    if required.temperature_regime is not None:
        stated = True
        if _is_room(observed) is not True:
            return Compat.INCOMPATIBLE, "not a room-temperature value"
        levels.append(Compat.COMPATIBLE)
    if required.environment is not None or required.medium is not None:
        stated = True
        if required.environment is not None and observed.environment is not required.environment:
            return Compat.INCOMPATIBLE, (
                f"environment {observed.environment} ≠ required {required.environment}"
                if observed.environment
                else "exposure environment not reported"
            )
        if required.medium is not None and observed.medium != required.medium:
            if required.environment is None:
                return Compat.INCOMPATIBLE, "different medium"
            levels.append(Compat.COMPATIBLE)
        elif required.medium is None and required.environment is not None:
            levels.append(Compat.COMPATIBLE)
        else:
            levels.append(Compat.EXACT)
    if required.exposure_duration is not None:
        stated = True
        if observed.exposure_duration is None:
            return Compat.INCOMPATIBLE, "exposure duration not reported"
        level = _close(
            observed.exposure_duration, required.exposure_duration, 0.01, 0.10, relative=True
        )
        if level is Compat.INCOMPATIBLE:
            return level, "different exposure duration"
        levels.append(level)
    if required.relative_humidity is not None:
        stated = True
        if observed.relative_humidity is None:
            return Compat.INCOMPATIBLE, "relative humidity not reported"
        level = _close(
            Quantity(value=observed.relative_humidity.to("%"), unit="%"),
            Quantity(value=required.relative_humidity.to("%"), unit="%"),
            1.0,
            5.0,
        )
        if level is Compat.INCOMPATIBLE:
            return level, "different relative humidity"
        levels.append(level)
    if not stated:
        if _is_room(observed) is False:
            return (
                Compat.INCOMPATIBLE,
                "non-room-temperature value for an unconditioned requirement",
            )
        return Compat.UNCONDITIONED, "general value"
    level = min(levels)
    return (
        level,
        "all stated conditions match" if level is Compat.EXACT else "equivalent conditions",
    )
