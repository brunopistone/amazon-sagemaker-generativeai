"""Name -> reward-function registry with file-based auto-discovery.

Each reward lives in its own module in this package and registers itself with a
decorator, so dropping a new file into the directory makes a new
``--reward_funcs`` name available without editing the training script.

Two decorators, depending on whether the reward takes a per-run hyperparameter:

  ``@register_reward("format")``          the decorated function *is* the reward.
  ``@register_reward_factory("length")``  the decorated function *builds* the
                                         reward from the run's script arguments.

Both return the decorated object unchanged. That is deliberate: TRL derives the
logged metric name from ``reward_func.__name__`` (``rewards/<name>/mean``), so
wrapping a reward in ``functools.wraps``-less glue - or in a ``functools.partial``,
which has no ``__name__`` at all - would rename or break its metrics.
"""

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# Reward name -> builder(script_args) -> reward callable.
# Plain rewards are stored as a builder that ignores its argument, so resolution
# has a single code path.
_BUILDERS: Dict[str, Callable[[Any], Callable]] = {}

# Module name -> exception, for reward modules that failed to import. Recorded
# rather than raised so one broken file in the directory cannot take down a run
# that does not use it; surfaced in the error message if a name cannot resolve.
_IMPORT_ERRORS: Dict[str, Exception] = {}

_discovered = False


def _register(name: str, builder: Callable[[Any], Callable]) -> None:
    if not name or ":" in name or "," in name:
        raise ValueError(
            f"Invalid reward name {name!r}: must be non-empty and contain neither "
            "',' (the --reward_funcs separator) nor ':' (the external-import marker)."
        )
    if name in _BUILDERS:
        raise ValueError(
            f"Reward name {name!r} is already registered. Every reward in "
            "this package needs a unique name."
        )
    _BUILDERS[name] = builder


def register_reward(name: str) -> Callable:
    """Register the decorated function itself as the reward called ``name``."""

    def decorator(func: Callable) -> Callable:
        _register(name, lambda script_args: func)
        return func

    return decorator


def register_reward_factory(name: str) -> Callable:
    """Register the decorated function as a *builder* for the reward ``name``.

    The builder is called once per run with the parsed script arguments and must
    return the reward callable. Use it for a reward that depends on a config
    value: read the value off ``script_args`` (with a sensible default so the
    builder still works when called with ``None``) and close over it.
    """

    def decorator(factory: Callable) -> Callable:
        _register(name, factory)
        return factory

    return decorator


def discover(package_name: str, package_paths: List[str]) -> None:
    """Import every public module in this package so decorators run.

    ``package_paths`` comes from the package's own ``__path__``, which Python
    resolves from the location of the package on disk - not from the current
    working directory. Discovery therefore behaves identically whether the
    training script is launched by ``python``, ``torchrun``, ``accelerate
    launch``, or a SageMaker entry point, and from any CWD.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    for module_info in pkgutil.iter_modules(package_paths):
        if module_info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package_name}.{module_info.name}")
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            _IMPORT_ERRORS[module_info.name] = e
            logger.warning(
                f"Could not import reward module '{module_info.name}': {e!r}. "
                "Any reward it defines will be unavailable."
            )


def available_rewards() -> List[str]:
    """Names registered by the reward modules in this package."""
    return sorted(_BUILDERS)


def build_reward(name: str, script_args: Any = None) -> Callable:
    """Instantiate the registered reward ``name`` for this run."""
    func = _BUILDERS[name](script_args)
    if not callable(func):
        raise TypeError(
            f"The builder registered for reward {name!r} returned "
            f"{type(func).__name__}, not a callable."
        )
    return func


def is_registered(name: str) -> bool:
    return name in _BUILDERS


def import_errors() -> Dict[str, Exception]:
    return dict(_IMPORT_ERRORS)
