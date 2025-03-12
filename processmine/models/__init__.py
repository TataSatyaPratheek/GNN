# processmine/models/__init__.py
"""
Model implementations for process mining.
"""
from processmine.models.base import BaseModel

# Import factory function
def create_model(model_type, **kwargs):
    """
    Factory function to create models with consistent interface
    
    Args:
        model_type: Type of model ('gnn', 'lstm', etc.)
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    from processmine import create_model as root_create_model
    return root_create_model(model_type, **kwargs)
