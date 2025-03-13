import sys
import os
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing processmine package
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)

# Log the path configuration for debugging
logger.info(f"Added path for imports: {package_path}")
logger.info(f"Current sys.path: {sys.path}")

# Try importing the package to verify it's accessible
try:
    import processmine
    logger.info(f"Successfully imported processmine package: {processmine.__file__}")
except ImportError as e:
    logger.warning(f"Could not import processmine package: {str(e)}")

# The PyTorch Geometric import check is no longer needed
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    logger.warning("PyTorch not installed")

try:
    import dgl
    logger.info(f"DGL version: {dgl.__version__}")
except ImportError:
    logger.warning("DGL not installed")