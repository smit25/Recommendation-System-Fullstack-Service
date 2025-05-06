#!/bin/bash

VERSION_FILE="version.yaml"

# Extract the current app version
APP_VERSION=$(grep "current_version:" $VERSION_FILE | awk '{print $2}')
if [ -z "$APP_VERSION" ]; then
    echo "Error: APP_VERSION not found in $VERSION_FILE"
    exit 1
fi

# Export APP_VERSION so Docker Compose can use it
export APP_VERSION
echo "Deploying with version: ${APP_VERSION}"

# Check if any flask-app containers are running
EXISTING_CONTAINERS=$(docker ps --format "{{.Names}}" | grep 'flask-app-' | wc -l)

if [ "$EXISTING_CONTAINERS" -eq 0 ]; then
    echo "First-time deployment detected! Creating two Flask app instances..."

    # Start Flask App 1
    echo "Starting flask-app-1-v${APP_VERSION}..."
    docker compose up -d --no-deps --force-recreate flask-app-1

    # Start Flask App 2
    echo "Starting flask-app-2-v${APP_VERSION}..."
    docker compose up -d --no-deps --force-recreate flask-app-2

    # Start NGINX Load Balancer
    echo "Starting Load Balancer..."
    docker compose up -d --no-deps --force-recreate load-balancer

    echo "First-time deployment complete!"

    # Start Monitoring tools
    echo "Starting prometheus"
    docker compose up -d --no-deps --force-recreate prometheus
    echo "Starting grafana"
    docker compose up -d --no-deps --force-recreate grafana
else
    echo "Rolling update: Replacing the oldest Flask container..."

    # Find the oldest running flask-app container
    OLDEST_CONTAINER=$(docker ps --format "{{.Names}}" | grep 'flask-app-' | sort -k2 | head -n 1 | awk '{print $1}')

    if [ -z "$OLDEST_CONTAINER" ]; then
        echo "Error: No valid Flask app container found!"
        exit 1
    fi

    echo "Stopping and removing $OLDEST_CONTAINER..."
    docker stop $OLDEST_CONTAINER && docker rm $OLDEST_CONTAINER

    # Determine which app type to replace (1 or 2)
    if [[ $OLDEST_CONTAINER == *"flask-app-1"* ]]; then
        APP_TYPE="flask-app-1"
    else
        APP_TYPE="flask-app-2"
    fi

    # Deploy a new container with updated version
    echo "Deploying new container for ${APP_TYPE} with version ${APP_VERSION}..."
    docker compose up -d --no-deps --force-recreate $APP_TYPE

    echo "Update complete!"
fi

# Show running containers
docker ps --format "table {{.Names}}\t{{.Status}}"