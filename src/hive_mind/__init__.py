"""Hive Mind - Multi-model conversation framework with collective intelligence."""

__version__ = "0.1.0"
__author__ = "Hive Mind Team"

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "HiveMind":
        from .core.hive_mind import HiveMind
        return HiveMind
    elif name == "Conversation":
        from .core.conversation import Conversation
        return Conversation
    elif name == "ModelResponse":
        from .models.response import ModelResponse
        return ModelResponse
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["HiveMind", "Conversation", "ModelResponse"]