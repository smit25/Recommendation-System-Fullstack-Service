#!/bin/bash

set -e

APP_NAME="flask-app"
NEW_VERSION="${APP_VERSION}"
OLD_VERSION="${OLD_APP_VERSION}"

NEW_CONTAINER_1="${APP_NAME}-1-v${NEW_VERSION}"
NEW_CONTAINER_2="${APP_NAME}-2-v${NEW_VERSION}"
PREVIOUS_CONTAINER_1="${APP_NAME}-1-v${OLD_VERSION}"
PREVIOUS_CONTAINER_2="${APP_NAME}-2-v${OLD_VERSION}"

echo "Deploying version: $NEW_VERSION"

# Export for docker-compose use
export APP_VERSION="$NEW_VERSION"

# Start new containers
docker-compose up -d --no-deps --build "$NEW_CONTAINER_1" "$NEW_CONTAINER_2"

echo "New containers started: $NEW_CONTAINER_1 and $NEW_CONTAINER_2"
echo "Waiting for health checks..."
sleep 30  # Allow time for initialization

# Check container health
check_health() {
  local container=$1
  docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unhealthy"
}

health1=$(check_health "$NEW_CONTAINER_1")
health2=$(check_health "$NEW_CONTAINER_2")

if [[ "$health1" != "healthy" || "$health2" != "healthy" ]]; then
  echo "Health check failed: $NEW_CONTAINER_1 ($health1), $NEW_CONTAINER_2 ($health2)"
  echo "Rolling back to previous containers..."

  docker rm -f "$NEW_CONTAINER_1" "$NEW_CONTAINER_2" || true

  echo "Restarting previous containers: $PREVIOUS_CONTAINER_1, $PREVIOUS_CONTAINER_2"
  docker start "$PREVIOUS_CONTAINER_1" || echo "Failed to restart $PREVIOUS_CONTAINER_1"
  docker start "$PREVIOUS_CONTAINER_2" || echo "Failed to restart $PREVIOUS_CONTAINER_2"

  echo "Deployment failed. Rolled back to previous version."
  exit 1
fi

# Cleanup old containers
echo "Removing old containers..."
docker rm -f "$PREVIOUS_CONTAINER_1" "$PREVIOUS_CONTAINER_2" || echo "Old containers not found or already removed."

# Cleanup old images
echo "Removing old images..."
OLD_IMAGE="${APP_NAME}-${OLD_VERSION}"
docker rmi "$OLD_IMAGE" || true

echo "Deployment complete and containers are healthy."