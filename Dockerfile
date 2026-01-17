FROM nvidia/cuda:11.2.2-base-ubuntu20.04
LABEL authors="victor"

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && apt-get clean

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio

RUN pip3 install transformers datasets tqdm

# EXPOSE 5000

# Run train
CMD ["python3", "main.py"]
