#!/bin/bash
# model_training_cron.sh - Automated model training without service disruption

# Configuration
LOG_FILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"
VERSION_FILE="version.yaml"
VERSION_UPDATE_TYPE="minor"  # Options: major, minor, patch
TRAINING_ENDPOINT="http://localhost:8085/train"  # Your load balancer endpoint
MODEL_DIR="/var/lib/jenkins/workspace/mlip-model-train/service/model"  # Directory where models are stored

# Create log directory if it doesn't exist
mkdir -p ./logs

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "========== Model Training Pipeline Started: $(date) =========="

# Function for error handling
handle_error() {
    echo "ERROR: $1"
    echo "========== Pipeline Failed: $(date) =========="
    exit 1
}

# Step 1: Generate new model version name
DATE_SUFFIX=$(date +%Y%m%d)
NEW_MODEL_VERSION="svd_model_${DATE_SUFFIX}"
echo "Using new model version: ${NEW_MODEL_VERSION}"

# Step 2: Trigger model training with specific version name
echo "Starting model training for version ${NEW_MODEL_VERSION}..."
TRAINING_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"version\": \"${NEW_MODEL_VERSION}\", \
         \"params\": { \
            \"n_factors\": 100, \
            \"n_epochs\": 20, \
            \"lr_all\": 0.005, \
            \"reg_all\": 0.02 \
         }}" \
    $TRAINING_ENDPOINT)

# Check if training started successfully
if [[ "$TRAINING_RESPONSE" != *"success"* ]]; then
    handle_error "Failed to start model training: $TRAINING_RESPONSE"
fi

echo "Training initiated successfully. Waiting for training to complete..."

# Wait for training to complete (adjust time based on your typical training duration)
sleep 120  # 2 minutes - adjust based on training time

# Verify the model file exists
if [ ! -f "${MODEL_DIR}/${NEW_MODEL_VERSION}.pkl" ] && [ ! -f "${MODEL_DIR}/${NEW_MODEL_VERSION//./-}.pkl" ]; then
    handle_error "Model file not found after training. Training may have failed."
fi

# Step 3: Update version in version.yaml
echo "Updating version in ${VERSION_FILE}..."
./increment_version.sh $VERSION_UPDATE_TYPE || handle_error "Failed to update version"

echo "========== Cron Job Completed Successfully: $(date) =========="
# # Step 4: Trigger rolling update for first container
# echo "Triggering rolling update for first container..."
# ./rolling_update.sh || handle_error "Failed to update first container"

# # Wait for first container to stabilize
# echo "Waiting for first container to stabilize..."
# sleep 30

# # Step 5: Trigger rolling update for second container
# echo "Triggering rolling update for second container..."
# ./rolling_update.sh || handle_error "Failed to update second container"

# echo "========== Pipeline Completed Successfully: $(date) =========="