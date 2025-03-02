# Use the base image you're working with, e.g., Python with specific versions if needed
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set up working directory
WORKDIR /workspace

# Set up build arguments for Git commit, version, and access token
ARG GIT_COMMIT
ARG VERSION_TAG

# Add metadata to the image
LABEL git_commit=${GIT_COMMIT}
LABEL version=${VERSION_TAG}

# Store version info in a file accessible within the container
RUN echo "git_commit=${GIT_COMMIT}" > image-info.txt && \
    echo "version=${VERSION_TAG}" >> image-info.txt

# Install any system dependencies and clean up in a single RUN command
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    g++ \
    libgl1-mesa-glx \
    wget \
    git \
    curl \
    nano \
    supervisor \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git lfs install

# setup cloudflared
RUN mkdir -p --mode=0755 /usr/share/keyrings \ 
    && curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null \ 
    && echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | tee /etc/apt/sources.list.d/cloudflared.list \ 
    && apt-get update \ 
    && apt-get install -y cloudflared \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up InstantID and install its dependencies
RUN git clone https://github.com/photoangell/InstantID.git \
    && if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi \
    && if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi \
    && python3 -m pip install --upgrade pip \
    && pip install -r InstantID/gradio_demo/requirements.txt \
    #&& pip install pickleshare opencv-python openai \
    && pip cache purge

# Expose the Jupyter, gradio and flask port
EXPOSE 8080 7860 5000

COPY supervisord.conf /etc/supervisord.conf
#CMD ["supervisord", "-c", "/etc/supervisord.conf"]
