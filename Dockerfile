FROM python:3.9-bullseye

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 ffmpeg wget curl git libsndfile1

WORKDIR /app

ARG CACHE_DIR=/root/.cache/pip

RUN pip install -U pip setuptools numpy

COPY requirements.txt ./

RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements.txt

RUN rm -rf /var/cache/apt/*

COPY . .

EXPOSE 6870
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-c", "/app/entrypoint.sh"]