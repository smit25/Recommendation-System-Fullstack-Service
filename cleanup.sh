#!/bin/bash

# List all containers with names that contain 'flask-app-'
FLASK_CONTAINERS=$(docker ps --format "{{.Names}}" | grep 'flask-app-')

# Check if any Flask containers are running
if [ -z "$FLASK_CONTAINERS" ]; then
  echo "No Flask containers found. Nothing to clean up."
  exit 0
fi

# Loop through and stop/remove all Flask containers
for CONTAINER in $FLASK_CONTAINERS; do
  echo "Stopping and removing Flask container: $CONTAINER"
  docker stop $CONTAINER
  docker rm $CONTAINER
done

echo "Cleanup complete!"
