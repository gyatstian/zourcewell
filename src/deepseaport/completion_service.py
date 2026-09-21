"""Backward-compat alias: real module lives in deepseaport.providers.deepseek.completion_service."""
import sys as _sys
from .providers.deepseek import completion_service as _impl
_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)