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

RUN ln -s /usr/bin/source-extractor /usr/local/bin/sex

COPY . /app
RUN pip3 install .