"""Backward-compat alias: real module lives in deepseaport.core.transport."""
import sys as _sys
from .core import transport as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)