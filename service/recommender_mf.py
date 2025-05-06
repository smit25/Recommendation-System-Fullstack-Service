from model.matrix_factorization import MatrixFactorization
from utils.sql import get_watched_movies, log_prediction
from utils.mlflow_config import setup_mlflow, safe_mlflow, REQUEST_SAMPLING_RATE
from time import time
import threading
import mlflow
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, version: str = "svd_model", pipeline_version: str = "1.0", experiment_name: str = "movie_recommendations_service"):
        """Initialize the MovieRecommender"""
        # Configure MLflow
        self.mlflow_available = setup_mlflow(experiment_name)
        
        # Only log service startup if MLflow is available
        if self.mlflow_available:
            self._log_service_startup(version)
            
        # Load model (no MLflow dependency for core functionality)
        self.model = MatrixFactorization(version, pipeline_version, experiment_name=experiment_name)
        
        # Initialize counters for sampled metrics
        self.request_count = 0
        self.last_batch_time = time()
        self.response_times = []
        
        logger.info(f"MovieRecommender initialized with model version {version}")
    
    @safe_mlflow
    def _log_service_startup(self, version):
        """Log service startup event"""
        with mlflow.start_run(run_name="service_startup"):
            mlflow.log_params({
                "model_version": version,
                "startup_time": datetime.now().isoformat(),
                "service_type": "movie_recommender"
            })
    
    def recommend(self, user_id, n=20, engine=None):
        """Recommend movies with minimal overhead"""
        # Start timing for potential sampling
        start_time = time()
        
        # Get recommendations
        watched = get_watched_movies(user_id, engine)
        recommendations = self.model.recommend(user_id, watched, n)
        
        # Calculate response time
        response_time = time() - start_time
        
        # Increment counter
        self.request_count += 1
        
        # Sample a tiny fraction of requests for detailed metrics
        if self.mlflow_available and random.random() < REQUEST_SAMPLING_RATE:
            self.response_times.append(response_time)
        
        # Check if it's time to log batch metrics
        current_time = time()
        if (self.mlflow_available and 
            (self.request_count >= 100000 or 
             (current_time - self.last_batch_time > 3600))):  # 1 hour
            
            threading.Thread(target=self._log_batch_metrics).start()
            self.request_count = 0
            self.last_batch_time = current_time
            self.response_times = []
        
        # Use existing logging system
        threading.Thread(target=log_prediction, args=(
            self.model.version,
            self.model.pipeline_version,
            self.model.start_date,
            self.model.end_date,
            user_id,
            recommendations,
            engine
        )).start()
        
        return recommendations
    
    @safe_mlflow
    def _log_batch_metrics(self):
        """Log aggregated metrics from a batch of requests"""
        with mlflow.start_run(run_name="batch_metrics"):
            # Log aggregate metrics
            mlflow.log_metric("total_requests", self.request_count)
            
            # Log sampled response time statistics if available
            if self.response_times:
                avg_time = sum(self.response_times) / len(self.response_times)
                mlflow.log_metric("avg_response_time", avg_time)
                if len(self.response_times) > 1:
                    sorted_times = sorted(self.response_times)
                    mlflow.log_metric("p50_response_time", sorted_times[len(sorted_times)//2])
                    if len(sorted_times) >= 20:  # Only log percentiles if we have enough samples
                        mlflow.log_metric("p95_response_time", sorted_times[int(len(sorted_times)*0.95)])
                        mlflow.log_metric("p99_response_time", sorted_times[int(len(sorted_times)*0.99)])
                    mlflow.log_metric("max_response_time", max(self.response_times))
                    mlflow.log_metric("min_response_time", min(self.response_times))