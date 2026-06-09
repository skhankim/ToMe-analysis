FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    rsync \
    openssh-client \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ToMe

COPY requirements.txt .

RUN pip install --upgrade "pip<25" "setuptools<70" wheel \
    && pip install --prefer-binary -r requirements.txt

COPY . .

RUN pip install -e ToMe-main/

CMD ["bash"]
