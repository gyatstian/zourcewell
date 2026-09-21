"""Backward-compat alias: real module lives in deepseaport.providers.deepseek.pow."""
import sys as _sys
from .providers.deepseek import pow as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)