# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app


RUN apt-get update 

RUN apt-get install -y \
    build-essential \
    software-properties-common \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip3 install streamlit==1.23.1

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Home.py", "localhost"]
