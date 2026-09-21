"""Backward-compat alias: real module lives in deepseaport.core.tools."""
import sys as _sys
from .core import tools as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)