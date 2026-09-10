"""Environment-variable helpers with legacy fallback.

Older versions of this project used the ``LOCALCREW_`` prefix; the
package has since been renamed to ``herding-cats``. To keep upgrade
paths painless, every public env-var lookup goes through
:func:`get_env`, which tries the new ``HERDING_CATS_*`` name first
and falls back to the legacy ``LOCALCREW_*`` name.

New code should read env vars via this module. The fallback chain is
documented in the project README under "Configuration".
"""

from __future__ import annotations

import os
from typing import Final

# Legacy prefix used before the rename. Kept as a public constant so
# callers / docs can reference it.
LEGACY_PREFIX: Final = "LOCALCREW_"


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable with the legacy ``LOCALCREW_*`` fallback.

    Lookup order:

    1. ``HERDING_CATS_<NAME>`` (the new prefix; canonical)
    2. ``LOCALCREW_<NAME>`` (the old prefix; deprecated but supported)
    3. ``default``

    The ``name`` argument is the *suffix* after the prefix, e.g.
    ``get_env("OLLAMA_URL")`` checks ``HERDING_CATS_OLLAMA_URL`` and
    then ``LOCALCREW_OLLAMA_URL``.
    """
    new_key = f"HERDING_CATS_{name}"
    legacy_key = f"{LEGACY_PREFIX}{name}"
    value = os.environ.get(new_key)
    if value is not None:
        return value
    return os.environ.get(legacy_key, default)
