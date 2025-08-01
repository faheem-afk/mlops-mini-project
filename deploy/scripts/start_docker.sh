#!/bin/bash

# login into aws ecr

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 596514762357.dkr.ecr.us-east-1.amazonaws.com
docker pull 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:v3

if ["$(docker ps -q -f name=my-container)"]; then
    docker stop my-container
    docker rm my-container
fi

docker run -d -p 80:5000 --name my-container 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:v3