FROM python:3.10-bullseye

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 ffmpeg wget curl git

WORKDIR /app

RUN pip install scikit-learn

COPY requirements.txt ./

RUN pip install -r requirements.txt 

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

EXPOSE 6870

CMD python UVR-webui.py