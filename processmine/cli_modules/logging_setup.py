#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging setup for ProcessMine CLI
"""

import os
import logging
import warnings
import sys
from typing import Dict, Any, Optional

def setup_logging(args: Any = None) -> logging.Logger:
    """
    Setup logging and suppress warnings
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured logger
    """
    # Create root logger
    logger = logging.getLogger("processmine")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Suppress common warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Precision is ill-defined")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    
    # For PyTorch warnings
    import torch
    torch.set_printoptions(precision=8)  # Prevent scientific notation warnings
    
    # For NumPy
    import numpy as np
    np.set_printoptions(precision=8, suppress=True)  # Prevent scientific notation warnings
    
    # For Tensorflow (if used)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        pass
    
    # Set lower verbosity for common libraries
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("sklearn").setLevel(logging.ERROR)
    
    return logger