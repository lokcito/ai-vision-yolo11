FROM python:latest

# Instala dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV DEBUG FALSE
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

COPY . .
RUN mkdir /tmp/Ultralytics
RUN mkdir /tmp/Ultralytics/models/
RUN pip install ultralytics opencv-python
COPY ./models/yolo11n.pt /tmp/Ultralytics/models/yolo11n.pt
COPY ./config/settings.json /tmp/Ultralytics/settings.json
COPY ./config/persistent_cache.json /tmp/Ultralytics/persistent_cache.json
#RUN pip install -r requirements.txt
