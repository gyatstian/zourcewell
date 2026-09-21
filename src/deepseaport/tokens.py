"""Backward-compat alias: real module lives in deepseaport.core.tokens."""
import sys as _sys
from .core import tokens as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)