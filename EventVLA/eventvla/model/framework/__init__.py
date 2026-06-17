from eventvla.model.tools import FRAMEWORK_REGISTRY
from eventvla.model.framework.EventVLA import EventVLA
        
def build_framework(cfg):
    """Build the public EventVLA model."""

    if not hasattr(cfg.framework, "name"):
        raise ValueError("Missing required config field `framework.name`.")
        
    if cfg.framework.name != "EventVLA":
        raise NotImplementedError("EventVLA open-source build only supports framework.name=EventVLA.")

    return EventVLA(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
