import math, numpy as np

def _json_sanitize(x):
    """Rekurencyjnie zamienia NaN/Inf -> None i numpy typy -> natywne py."""
    if x is None:
        return None
    if isinstance(x, (str, bool, int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    # fallback – nieznany typ
    return str(x)