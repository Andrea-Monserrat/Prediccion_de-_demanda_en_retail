#!/usr/bin/env bash

image=$1

if [ "$image" == "" ]; then
    echo "Uso: $0 <nombre-imagen>"
    exit 1
fi

chmod +x retail_forecast/train
chmod +x retail_forecast/serve

account=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]; then
    exit 255
fi

region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

aws ecr get-login-password --region "${region}" \
    | docker login --username AWS --password-stdin \
      "${account}.dkr.ecr.${region}.amazonaws.com"

docker build --network sagemaker -t "${image}" .
docker tag "${image}" "${fullname}"
docker push "${fullname}"
