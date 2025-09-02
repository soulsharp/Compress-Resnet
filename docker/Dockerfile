FROM python:3.13-slim
LABEL maintainer="soulsharp"
LABEL repository="Compress-Resnet"

ENV DEBIAN_FRONTEND=noninteractive

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Creates missing files that werent copied
RUN mkdir /data/test /model/weights

# Container paths to datasets
ENV TEST_DATASET_PATH="/data/test"

RUN python -m pip install --no-cache-dir --upgrade pip 
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt