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
EXPOSE 5000
RUN pip3 install -r requirements.txt
ENTRYPOINT mlflow run /app/ -P n_folds=5 --run-name=abtest --env-manager=local && mlflow ui