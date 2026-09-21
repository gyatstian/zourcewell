"""Backward-compat alias: real module lives in deepseaport.core.obscura_bridge."""
import sys as _sys
from .core import obscura_bridge as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)