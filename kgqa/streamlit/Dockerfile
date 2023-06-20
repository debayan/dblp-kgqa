# app/Dockerfile

FROM python:3.10

WORKDIR /app

COPY . .

#RUN apt-get install -y \
#    build-essential \
#    software-properties-common \ 
#    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip3 install streamlit

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "home.py"]
