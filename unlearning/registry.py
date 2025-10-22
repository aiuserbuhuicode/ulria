"""Simple registry for machine unlearning algorithms.

Algorithms register themselves by name under the ``"mu"`` category.  The
``REGISTRY`` mapping can then be used by the CLI to instantiate the
appropriate class based on the ``--mu`` command‑line argument.
"""

from __future__ import annotations

from typing import Dict, Type

REGISTRY: Dict[str, Dict[str, Type]] = {"mu": {}}


def register_mu(name: str):
    """Decorator used by unlearning algorithms to register themselves."""

    def decorator(cls):
        REGISTRY["mu"][name] = cls
        return cls

    return decorator


try:
    from . import sisa  # noqa: F401
    from . import fisher  # noqa: F401
    from . import kd  # noqa: F401
    from . import prune_class  # noqa: F401
    from . import retrain  # noqa: F401
except Exception:
    # Fail gracefully if optional dependencies are missing
    pass
