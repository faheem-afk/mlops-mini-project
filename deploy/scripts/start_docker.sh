#!/bin/bash

# login to aws ecr

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 596514762357.dkr.ecr.us-east-1.amazonaws.com
docker pull 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:latest
docker stop my-container || true
docker rm my-container || true
docker run -p 80:5000 --name my-container 596514762357.dkr.ecr.us-east-1.amazonaws.com/mlops-mini-project:latest