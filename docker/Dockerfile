FROM python:3.13-slim
LABEL maintainer="soulsharp"
LABEL repository="Compress-Resnet"

RUN apt-get update && \ 
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends\
    build-essential libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install -r requirements.txt

COPY . .

# Creates missing files that werent copied
RUN mkdir -p /data/test /model/weights

# Container paths to datasets
ENV TEST_DATASET_PATH="/data/test"
ENV MODEL_WT_PATH="/model/weights"

CMD ["python", "-m", "scripts.run_benchmark"]