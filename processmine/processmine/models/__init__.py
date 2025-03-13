# processmine/models/__init__.py
"""
Model implementations for process mining.
"""
from processmine.models.base import ProcessModel
from processmine.models.factory import create_model, get_model_config

# Re-export factory functions for convenience
__all__ = ['ProcessModel', 'create_model', 'get_model_config']