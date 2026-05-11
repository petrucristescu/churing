#!/bin/bash
set -e

IMAGE="ghcr.io/petrucristescu/churing:latest"

if ! command -v docker &>/dev/null; then
  echo "Docker is required: https://docs.docker.com/get-docker/"
  exit 1
fi

echo "Pulling latest Churing..."
docker pull "$IMAGE"
echo ""
docker run -it --rm "$IMAGE"
