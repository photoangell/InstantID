# Use the base image you're working with, e.g., Python with specific versions if needed
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set up working directory
WORKDIR /workspace

# Install any system dependencies and clean up in a single RUN command
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    g++ \
    libgl1-mesa-glx \
    wget \
    git \
    curl \
    nano \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git lfs install

# Set up InstantID and install its dependencies
RUN git clone https://github.com/photoangell/InstantID.git \
    && if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi \
    && if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi \
    && python3 -m pip install --upgrade pip \
    && pip install -r InstantID/gradio_demo/requirements.txt \
    && pip install jupyter pickleshare mediapipe \
    && pip install --upgrade huggingface-hub diffusers \
    && pip cache purge

# Expose the Jupyter & gradio port
EXPOSE 8080 7860

# Run Jupyter on container startup with the custom token - this is not required when running in Vast.Ai
# This command is in the vast.ai template
#CMD ["venv-dev/bin/jupyter", "notebook", "--ServerApp.token='YmbbtWillBlowYourTinyMind'", "--port=8080", "--no-browser", "--allow-root"]
