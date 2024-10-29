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

```bash
docker build -t photoangell/ymbbt-faceclone-dev:latest .
```

### Running the Docker Container

To run the Docker container, use the following command:

```bash
docker run --gpus all -p 8080:8080 -p 7860:7860 photoangell/ymbbt-faceclone-dev:latest
```

## Input and Output volumes with a python watcher script
Input Folder: /data/input
Output Folder: /data/output

```bash
docker run -v /host/input:/data/input -v /host/output:/data/output photoangell/ymbbt-faceclone-dev:latest
```
### Orchestrating with Docker Restart Policies or Docker Compose

To handle any unexpected container stoppages, use a restart policy that ensures the container comes back up:

```bash
docker run --restart=always -v /host/input:/data/input -v /host/output:/data/output photoangell/ymbbt-faceclone-dev:latest
```

Alternatively, if you are using Docker Compose:
```yaml
version: '3'
services:
  image_processor:
    image: your_image
    volumes:
      - /host/input:/data/input
      - /host/output:/data/output
    restart: always
```