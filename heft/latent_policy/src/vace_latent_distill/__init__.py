"""Instruction-conditioned, past-only distillation of VACE spatial structure."""

from .config import ModelConfig, PipelineConfig, TeacherConfig, TrainConfig

__all__ = ["ModelConfig", "PipelineConfig", "StructuredStudent", "TeacherConfig", "TrainConfig"]


def __getattr__(name):
    if name == "StructuredStudent":
        from .model import StructuredStudent

        globals()[name] = StructuredStudent
        return StructuredStudent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
