"""Writing things a human is expected to open.

An airframe file is hand-editable and version-controllable, which puts two requirements on the
dumper that PyYAML does not meet on its own.

The first is that solver results arrive full of `np.float64`, and PyYAML's safe dumper cannot
represent one — it raises `RepresenterError` part-way through, which loses the *whole file* rather
than one field. Coercing at each producer would work right up until the first producer that
forgets, so it happens here, at the one place everything is written through.

The second is that `0.30000000000000004` has no business appearing in a file someone is going to
read.
"""
from __future__ import annotations

from typing import Any

import yaml

#: Decimal places every persisted float is rounded to.
#:
#: This is not only cosmetic. The cache key is taken over the airframe, so if the file stored a
#: rounded value while the hash saw the unrounded one, saving an airframe would change its key —
#: orphaning every cached solve and making every recorded result look stale the instant the file
#: was reopened. `canonical()` below exists so the hash sees exactly what the file will hold.
FLOAT_PRECISION = 10


def canonical(value):
    """The payload as it will be persisted: floats rounded, numpy coerced.

    Hash this rather than the raw object. It also makes the key robust to float noise in general —
    two optimiser candidates differing in the fifteenth decimal are the same aircraft, and giving
    them separate cache entries would be a slow way of being wrong.
    """
    value = plain(value)
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if type(value) is float:
        return round(value, FLOAT_PRECISION)
    return value


def _represent_floats_readably(dumper: yaml.SafeDumper, value: float) -> yaml.ScalarNode:
    """Stop 0.30000000000000004 appearing in a file a human is expected to edit."""
    return dumper.represent_scalar(
        "tag:yaml.org,2002:float", repr(round(value, FLOAT_PRECISION))
    )


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(float, _represent_floats_readably)


def plain(value: Any) -> Any:
    """Coerce numpy scalars and arrays to plain Python, recursively."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    # `type(...) in` rather than isinstance: **np.float64 subclasses Python float**, so an
    # isinstance shortcut passes it straight through untouched and PyYAML still cannot represent
    # it. That is exactly how this bug survived the first attempt at the fix.
    if type(value) in (bool, str, int, float, type(None)):
        return value
    tolist = getattr(value, "tolist", None)  # numpy arrays and scalars both have it
    if callable(tolist):
        return plain(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return plain(item())
    return value


def dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.dump(plain(payload), Dumper=_Dumper, sort_keys=False, default_flow_style=False)


def load_yaml(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("expected a mapping at the top level of the file")
    return loaded
