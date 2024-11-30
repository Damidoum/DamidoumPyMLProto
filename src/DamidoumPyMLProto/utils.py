from typing import Any
import numpy as np


def is_valid_array(value: Any) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError("Data must be a numpy array")
    if not value.dtype in [np.float32, np.float64]:
        raise ValueError("Data must be of type float32 or float64")
    if not value.ndim == 2:
        raise ValueError("Data must be 2 dimensional array")
