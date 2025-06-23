# Use official NVIDIA PyTorch image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy local files to the container
COPY . .

# Install Python packages
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run your Flask server
CMD ["python3", "server.py"]
