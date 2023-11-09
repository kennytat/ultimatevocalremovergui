FROM python:3.10-bullseye

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 ffmpeg wget curl git

WORKDIR /app

RUN pip install -U setuptools scikit-learn

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY requirements.txt ./

RUN pip install -r requirements.txt 

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

RUN sed -i "s/ass={ass_path}/ass='{ass_path}'/g" UVR-webui.py

EXPOSE 6870

CMD python UVR-webui.py