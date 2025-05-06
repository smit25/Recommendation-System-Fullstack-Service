#!/bin/bash

VERSION_FILE="version.yaml"

# Extract the current version from version.yaml
CURRENT_VERSION=$(grep "current_version:" $VERSION_FILE | awk '{print $2}')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Determine which version part to increment
case "$1" in
    major) ((MAJOR++)); MINOR=0; PATCH=0 ;;
    minor) ((MINOR++)); PATCH=0 ;;
    patch|*) ((PATCH++)) ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

# Update version.yaml
sed -i "s/^current_version:.*/current_version: $NEW_VERSION/" $VERSION_FILE
sed -i "s/^previous_version:.*/previous_version: $CURRENT_VERSION/" $VERSION_FILE

# Update Dockerfile
sed -i "s/^ARG APP_VERSION=.*/ARG APP_VERSION=$NEW_VERSION/" Dockerfile

# Update Jenkinsfile
sed -i 's/^APP_VERSION=.*/APP_VERSION="'"$NEW_VERSION"'"/' Jenkinsfile

# Confirmation output
echo "Version updated to $NEW_VERSION"

# # Optional Git operations
# git add $VERSION_FILE Dockerfile Jenkinsfile
# git commit -m "Bump version to $NEW_VERSION"
# git tag "v$NEW_VERSION"
# git push origin main --tags
