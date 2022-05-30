FROM ubuntu:20.04

RUN apt-get update && apt-get install build-essential postgresql-client binutils libproj-dev  -y
RUN apt install cabextract -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y software-properties-common && apt-get --allow-releaseinfo-change update
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8 && apt-get install -y python3-pip
RUN apt-get install -y git

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN mkdir /app

# set the working directory to /mining
WORKDIR /app

# copy the current directory contents into the container at /mining
COPY requirements.txt requirements.txt
COPY . /app/
RUN ls -la /app/
RUN pip3 install -r requirements.txt
EXPOSE 8080

# CMD ["/bin/bash", "-c", "/app/setup.sh"]
# ENTRYPOINT streamlit run app.py --server.port
CMD streamlit run --server.port 8080 --server.enableCORS false app.py