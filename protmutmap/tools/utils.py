import numpy as np

def is_nan_or_none(value):
    """Check if value is np.nan or None"""
    return value is None or (isinstance(value, float) and np.isnan(value))
