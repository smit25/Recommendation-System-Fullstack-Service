import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader, accuracy
from utils.sql import get_ratings_df
from utils.mlflow_config import setup_mlflow, safe_mlflow, REQUEST_SAMPLING_RATE
import mlflow
import mlflow.pyfunc
import os
import logging
from datetime import datetime
import random
import threading

logger = logging.getLogger(__name__)

# Define a custom PyFunc model for MLflow to track Surprise models
class SVDWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, svd_model):
        self.svd_model = svd_model
        
    def predict(self, context, model_input):
        # Expect input as DataFrame with user_id and movie_id columns
        return [self.svd_model.predict(row['user_id'], row['movie_id']).est 
                for _, row in model_input.iterrows()]

class MatrixFactorization():
    def __init__(self, version: str = "svd_model", pipeline_version: str = "1.0",
                experiment_name: str = "movie_recommendations"):
        # Initialize MLflow (without failing if unavailable)
        self.mlflow_available = setup_mlflow(experiment_name)
        
        self.svd = SVD()
        self.model = None
        self.version = version
        self.pipeline_version = pipeline_version
        self.file_path = '/var/lib/jenkins/workspace/mlip-model-train/service/model/{}.pkl'.format(version.replace('.', '-')) # version - svd_model_1_2_3
        
        # Load the existing model without MLflow dependency
        self._load_model_without_logging(version)
        
        # Get the movie data needed for recommendations
        self.df_temp = get_ratings_df()
        self.movies = self.df_temp['movie_id'].unique().tolist()
        self.start_date = self.df_temp['timestamp'].min()
        self.end_date = self.df_temp['timestamp'].max()
        
        # Initialize aggregation metrics
        self._request_count = 0
        self._last_log_time = datetime.now()
        
        # Log model load in background to not block initialization
        if self.mlflow_available:
            threading.Thread(target=self._log_model_loaded).start()
    
    @safe_mlflow
    def _log_model_loaded(self):
        """Log model load event without blocking"""
        with mlflow.start_run(run_name=f"model_load_{self.version}"):
            mlflow.log_params({
                "model_version": self.version,
                "pipeline_version": self.pipeline_version,
                "load_time": datetime.now().isoformat(),
                "movies_count": len(self.movies)
            })
    
    def _load_model_without_logging(self, version):
        """Load model with robust error handling"""
        try:
            with open('./service/model/{}.pkl'.format(version.replace('.', '-')), 'rb') as f:
                self.model = pickle.load(f)
                logger.info(f"Model {version} loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Model file for version {version} not found, using untrained model.")
            with open('./service/model/svd_model.pkl'.format(version.replace('.', '-')), 'rb') as f:
                self.model = pickle.load(f)
                logger.info(f"Model {version} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self.svd  # Initialize with untrained model
    
    @safe_mlflow
    def train(self, df: pd.DataFrame, params: dict = None) -> str:
        """Train model with MLflow tracking (if available)"""
        try:
            # Process the dataframe to ensure it has the right format
            # Check if required columns exist
            required_columns = ['user_id', 'movie_id', 'rating']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in training data")
            
            # Convert to proper types
            training_df = df.copy()
            training_df['user_id'] = training_df['user_id'].astype(str)
            training_df['movie_id'] = training_df['movie_id'].astype(str)
            training_df['rating'] = training_df['rating'].astype(float)
            
            # Log data information
            logger.info(f"Training with {len(training_df)} ratings from {training_df['user_id'].nunique()} users on {training_df['movie_id'].nunique()} movies")
            
            # Continue with MLflow tracking if available
            if not self.mlflow_available:
                logger.warning("MLflow not available, training without tracking")
                return self._train_without_mlflow(training_df, params)
                
            with mlflow.start_run(run_name=f"training_{self.version}") as run:
                # Training logic
                if params is None:
                    params = {}
                
                mlflow.log_params({
                    "model_version": self.version,
                    "pipeline_version": self.pipeline_version,
                    "n_factors": params.get("n_factors", 100),
                    "n_epochs": params.get("n_epochs", 20),
                    "lr_all": params.get("lr_all", 0.005),
                    "reg_all": params.get("reg_all", 0.02),
                    "data_start_date": str(training_df['timestamp'].min()) if 'timestamp' in training_df.columns else "unknown",
                    "data_end_date": str(training_df['timestamp'].max()) if 'timestamp' in training_df.columns else "unknown",
                    "n_users": training_df['user_id'].nunique(),
                    "n_movies": training_df['movie_id'].nunique(),
                    "n_ratings": len(training_df)
                })
                
                # Update model parameters if provided
                if params:
                    self.svd = SVD(**params)
                    
                self.movies = training_df['movie_id'].unique().tolist()
                
                # Store timestamp info if available
                if 'timestamp' in training_df.columns:
                    self.start_date = training_df['timestamp'].min()
                    self.end_date = training_df['timestamp'].max()
                
                # Train model using only the three required columns
                reader = Reader(rating_scale=(1, 5))
                dataset = Dataset.load_from_df(training_df[required_columns], reader)
                trainset = dataset.build_full_trainset()
                self.svd.fit(trainset)
                self.model = self.svd
                
                # Log metrics from training
                if hasattr(self.svd, 'train_rmse_'):
                    mlflow.log_metric("train_rmse", self.svd.train_rmse_)
                    
                # Save model to MLflow
                self._save_to_mlflow()
                
                # Always save locally
                self.save_model()
                
                logger.info("Training complete")
                return run.info.run_id
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    def _train_without_mlflow(self, df, params=None):
        """Fallback training method when MLflow is unavailable"""
        try:
            # Update model parameters if provided
            if params:
                self.svd = SVD(**params)
            
            # Make sure dataframe has the right columns
            required_columns = ['user_id', 'movie_id', 'rating']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in training data")
            
            # Ensure data types are correct
            df['user_id'] = df['user_id'].astype(str)
            df['movie_id'] = df['movie_id'].astype(str)
            df['rating'] = df['rating'].astype(float)
            
            # Log data summary for debugging
            logger.info(f"Training data: {len(df)} rows, columns: {df.columns.tolist()}")
            logger.info(f"Data types: {df.dtypes}")
            
            # Store metadata
            self.movies = df['movie_id'].unique().tolist()
            if 'timestamp' in df.columns:
                self.start_date = df['timestamp'].min()
                self.end_date = df['timestamp'].max()
            
            # Train model
            reader = Reader(rating_scale=(1, 5))
            dataset = Dataset.load_from_df(df[required_columns], reader)  # Only use required columns
            trainset = dataset.build_full_trainset()
            self.svd.fit(trainset)
            self.model = self.svd
            
            # Save model locally
            self.save_model()
            logger.info("Training complete (without MLflow tracking)")
            return None
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def predict(self, user: int, item: str) -> float:
        """Make prediction"""
        return self.model.predict(user, item).est
    
    @safe_mlflow
    def evaluate(self, df: pd.DataFrame) -> float:
        """Evaluate model with MLflow logging (if available)"""
        # Calculate metrics regardless of MLflow availability
        testset = [(str(row['user_id']), str(row['movie_id']), row['rating']) for _, row in df.iterrows()]
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        # Log to MLflow if available
        if self.mlflow_available:
            with mlflow.start_run(run_name=f"evaluation_{self.version}"):
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_size", len(testset))
        
        return rmse

    def save_model(self) -> None:
        """Save model locally (always works regardless of MLflow)"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @safe_mlflow
    def _save_to_mlflow(self) -> None:
        """Save model to MLflow model registry (if available)"""
        if not self.mlflow_available:
            return
            
        wrapper = SVDWrapper(self.model)
        
        signature = mlflow.models.signature.infer_signature(
            pd.DataFrame({'user_id': ['1'], 'movie_id': ['1']}),
            pd.Series([3.5])
        )
        
        # Fix: Remove the duplicate artifact_path parameter
        mlflow.pyfunc.log_model(
            python_model=wrapper,
            artifact_path="models",  # Only specify this once
            registered_model_name=f"SVD_MovieRec_{self.version}",
            signature=signature
        )
        
        # Log the saved model file as a separate artifact
        if os.path.exists(self.file_path):
            mlflow.log_artifact(self.file_path)
    
    def recommend(self, user_id: str, watched: list[str], N: int=20):
        """Get recommendations with minimal overhead"""
        # Increment request counter
        self._request_count += 1
        
        # Generate recommendations
        predictions = []
        for movie_id in self.movies:
            if movie_id in watched:
                continue
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = [p for p, _ in predictions[:N]]
        
        # Sample a tiny fraction of requests for tracking
        if self.mlflow_available and random.random() < REQUEST_SAMPLING_RATE:
            threading.Thread(target=self._log_recommendation_sample, 
                            args=(user_id, watched, top_recommendations)).start()
            
        # Log aggregated stats periodically in background
        current_time = datetime.now()
        if (self.mlflow_available and 
            (self._request_count >= 10000 or 
             (current_time - self._last_log_time).total_seconds() > 3600)):
            threading.Thread(target=self._log_aggregated_metrics).start()
            self._request_count = 0
            self._last_log_time = current_time
        
        return top_recommendations
    
    @safe_mlflow
    def _log_recommendation_sample(self, user_id, watched, recommendations):
        """Log a sampled recommendation for analysis"""
        with mlflow.start_run(run_name=f"sample_recommendation"):
            mlflow.log_params({
                "user_id": user_id,
                "model_version": self.version,
                "watched_count": len(watched),
                "recommendations_count": len(recommendations),
                "sample_timestamp": datetime.now().isoformat()
            })
    
    @safe_mlflow
    def _log_aggregated_metrics(self):
        """Log aggregated metrics in background"""
        with mlflow.start_run(run_name="aggregated_metrics"):
            mlflow.log_metrics({
                "recommendation_requests": self._request_count,
                "timestamp": datetime.now().timestamp()
            })