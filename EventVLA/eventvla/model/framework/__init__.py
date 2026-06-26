from importlib import import_module

from eventvla.model.tools import FRAMEWORK_REGISTRY
from eventvla.model.framework.EventVLA import EventVLA


def _framework_name(cfg):
    framework_cfg = cfg.framework
    name = getattr(framework_cfg, "name", None)
    if name is None:
        name = getattr(framework_cfg, "framework_py", None)
    if name is None:
        raise ValueError("Missing required config field `framework.name`.")
    return name


def _ensure_optional_framework_registered(name: str) -> None:
    if name == "Pi05MEM":
        import_module("eventvla.model.framework.Pi05MEM")


def build_framework(cfg):
    """Build a public EventVLA framework by name."""

    name = _framework_name(cfg)
    if name == "EventVLA":
        return EventVLA(cfg)

    _ensure_optional_framework_registered(name)
    if name in FRAMEWORK_REGISTRY._registry:
        return FRAMEWORK_REGISTRY[name](cfg)

    available = ", ".join(sorted(["EventVLA", *FRAMEWORK_REGISTRY._registry]))
    raise NotImplementedError(f"Framework {name!r} is not included. Available: {available}")


__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
