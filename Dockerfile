FROM python:3.10-bullseye

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 ffmpeg wget curl git libsndfile1
WORKDIR /app
ARG CACHE_DIR=/root/.cache/pip

COPY requirements.txt ./

RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} numpy==1.26.4
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements.txt
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} --force-reinstall numpy==1.26.4

RUN rm -rf /var/cache/apt/*

COPY . .
RUN mv libcudnn* /usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib/ 
## libcudnn.so.8 libcudnn_adv_infer.so.8 libcudnn_cnn_infer.so.8 libcudnn_ops_infer.so.8

ENV LD_LIBRARY_PATH=/usr/lib:/usr/lib64:/usr/local/lib:/usr/local/lib64/:/usr/lib/x86_64-linux-gnu:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

EXPOSE 6870
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-c", "/app/entrypoint.sh"]