import os
import mlflow
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the user's home directory and create a path there
HOME_DIR = os.path.expanduser("~")
DEFAULT_TRACKING_URI = f"file:///var/lib/jenkins/workspace/mlip-model-train/mlflow"

# Another option: use a relative path from current directory
# import pathlib
# CURRENT_DIR = pathlib.Path().absolute()
# DEFAULT_TRACKING_URI = f"file://{CURRENT_DIR}/mlflow_data"

# Get sampling rate from environment or use default
REQUEST_SAMPLING_RATE = float(os.environ.get("MLFLOW_REQUEST_SAMPLING_RATE", "0.001"))

def setup_mlflow(experiment_name="movie_recommendations"):
    """Configure MLflow with appropriate tracking URI and error handling"""
    try:
        # Get tracking URI from environment or use default
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        
        # Create the directory if it doesn't exist
        dir_path = tracking_uri.replace("file://", "")
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensuring MLflow directory exists: {dir_path}")
        
        # Set the tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow configured with tracking URI: {tracking_uri}")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure MLflow: {e}")
        return False

def safe_mlflow(func):
    """Decorator to make MLflow operations fault-tolerant"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"MLflow operation failed: {e}")
            return None
    return wrapper