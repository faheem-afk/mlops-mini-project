#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 596514762357.dkr.ecr.us-east-1.amazonaws.com

# Pull the latest image
docker pull 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:latest

# Check if the container 'campusx-app' is running
if [ "$(docker ps -q -f name=my-container)" ]; then
    # Stop the running container
    docker stop my-container
    docker rm my-container
fi

# Run a new container
docker run -d -p 80:5000 --name my-container 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:latest
