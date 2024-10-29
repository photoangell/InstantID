# FaceCloneDocker

## Overview

FaceCloneDocker is a Dockerized application for face cloning using deep learning models. This repository contains the necessary files to build and run the Docker image for the application.

## Getting Started

### Prerequisites

- Docker installed on your machine
- Docker Hub account
- Nvidia GPU with 24gb vram

### Building the Docker Image

To build the Docker image, run the following command in the root directory of the repository:

```sh
docker build -t photoangell/ymbbt-faceclone-dev:latest .
```

### Running the Docker Container

To run the Docker container, use the following command:

```sh
docker run --gpus all -p 8080:8080 -p 7860:7860 photoangell/ymbbt-faceclone-dev:latest
```
