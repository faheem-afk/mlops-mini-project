#!/bin/bash

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 596514762357.dkr.ecr.us-east-1.amazonaws.com

# Pull the Docker image
docker pull 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:v3

# If a container with the name 'my-container' is running or exists, stop and remove it
if [ "$(docker ps -aq -f name=my-container)" ]; then
    docker stop my-container
    docker rm my-container
fi

# Run the container
docker run -d -p 80:5000 --name my-container 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:v3