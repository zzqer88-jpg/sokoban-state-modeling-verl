"""Temporarily seed Python and NumPy without leaking global RNG changes."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Iterator

import numpy as np


@contextmanager
def set_seed(seed: int) -> Iterator[None]:
    """Run a deterministic block and restore both global RNG states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
