from flask import Flask, request, jsonify, Response
import time
import threading
from utils.sql import get_engine, get_ratings_df
from service.recommender_mf import MovieRecommender
from utils.util_functions import read_versions_from_yaml, parse_version
from model.matrix_factorization import MatrixFactorization
from utils.mlflow_config import setup_mlflow, safe_mlflow
import mlflow
import os
import logging
# from prometheus import start_http_server, Summary, Counter, Gauge
from prometheus_client import start_http_server, Summary, Counter, Gauge
import threading
import random
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure MLflow tracking
mlflow_available = setup_mlflow("movie_recommendation_service")
logger.info(f"Flask app started with MLflow tracking available: {mlflow_available}")

# Initialize services
prev_version, curr_version = read_versions_from_yaml("version.yaml")
major, minor, patch = parse_version(curr_version)
major1, minor1, patch1 = parse_version(prev_version)
version_old= f"svd_model_{major1}_{minor1}_{patch1}"
version_new = f"svd_model_{major}_{minor}_{patch}"  # Default to timestamp version

# Get Pipeline version
pipeline_version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

movie_recommender_old = MovieRecommender(version=version_old, pipeline_version=pipeline_version)
movie_recommender_new = MovieRecommender(version=version_new, pipeline_version=pipeline_version)
engine = get_engine()


@app.route('/health', methods=["GET"])
def health():
    """Health check endpoint for Docker"""
    return jsonify({"status": "healthy"}), 200


# Metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Request latency')
RECOMMENDATION_ACCURACY = Gauge('model_accuracy', 'Accuracy of model', ['model'])

@app.route('/recommend/<int:user_id>', methods = ["GET"])
def recommend(user_id):
    """Get movie recommendations for a user"""
    start_time = time.time()
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "Invalid user ID, must be an integer"}), 400
    
    ## Get recommendations for user_id
    if random.random() < 0.6:
        recommendations = movie_recommender_old.recommend(user_id=user_id, n=20, engine=engine)
    else:
        recommendations = movie_recommender_new.recommend(user_id=user_id, n=20, engine=engine)
    
    # Log response time and result for operational monitoring
    response_time = int((time.time() - start_time) * 1000)
    log_entry = f"{time.time()},{user_id},recommendation request server, status 200, result: {recommendations}, {response_time}ms"
    logger.info(log_entry)
    # Return as CSV string
    RECOMMENDATION_ACCURACY.labels(model="matrix_factorization").set(0.9)
    # producer.send(KAFKA_TOPIC, log_entry)
    # producer.flush()

    csv_string = ",".join(recommendations)
    return Response(csv_string, mimetype='text/csv')

@app.route('/train', methods=["POST"])
def train_model():
    """Endpoint to trigger model training"""
    try:
        data = request.get_json()
        _, curr_version = read_versions_from_yaml("version.yaml")
        major, minor, patch = parse_version(curr_version)

        version = f"svd_model_{major}_{minor+1}_{patch}"  # Default to timestamp version
        pipeline_version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        params = data.get('params', None)
        logger.info(f"MODEL VERSION {version}")
        # Run training in a background thread to not block the server
        def train_in_background():
            try:
                # Make sure MLflow is configured in this thread
                setup_mlflow("movie_recommendation_training")
                
                # Create a new model instance for training
                logger.info(f"Starting training for model version {version}")
                training_model = MatrixFactorization(version=version, pipeline_version=pipeline_version)
                
                # Get training data
                df = get_ratings_df()
                
                # Train the model
                run_id = training_model.train(df, params)
                
                # If training successful, reload the production model
                if run_id:
                    global movie_recommender
                    movie_recommender = MovieRecommender(version=version)
                    logger.info(f"Model training completed with run_id: {run_id}")
                else:
                    logger.error("Model training failed")
            except Exception as e:
                logger.error(f"Error in background training: {e}")
        
        # Start training in background
        training_thread = threading.Thread(target=train_in_background)
        training_thread.daemon = True  # Make thread exit when main thread exits
        training_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Model training started in background",
            "version": version
        }), 202  # 202 Accepted
            
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/predict")
@REQUEST_LATENCY.time()
def predict():
    user_id = int(request.args.get("user_id", 1))
    REQUEST_COUNT.inc()
    predicted_value = movie_recommender.recommend(user_id=user_id, n=5, engine=engine)
    accuracy = 0.9
    RECOMMENDATION_ACCURACY.labels(model="matrix_factorization").set(accuracy)
    
    return jsonify({
        "user_id": user_id,
        "result": predicted_value
    })



if __name__ == '__main__':
    threading.Thread(target=lambda: start_http_server(9100, addr="0.0.0.0")).start()
    app.run(host="0.0.0.0", port=8085, debug=True, use_reloader=False)