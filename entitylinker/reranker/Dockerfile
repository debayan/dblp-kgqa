# app/Dockerfile

FROM python:3.9-slim

USER root

RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' \
      /etc/apt/apt.conf.d/docker-clean

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "infer.py"]
