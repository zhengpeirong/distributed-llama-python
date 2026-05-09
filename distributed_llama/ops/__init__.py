"""C accelerator extension for distributed-llama ops."""

try:
    from ._ops import *  # noqa: F401, F403
except ImportError:
    pass
