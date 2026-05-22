FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    sextractor \
    swarp \
    scamp \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    libcfitsio-dev \
    wcslib-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python3", "main.py"]