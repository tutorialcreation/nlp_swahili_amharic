FROM ubuntu:20.04

RUN apt-get update && apt-get install build-essential postgresql-client binutils libproj-dev  -y
RUN apt-get install -y software-properties-common && apt-get --allow-releaseinfo-change update
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8 && apt-get install -y python3-pip
RUN apt-get install -y git
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
# wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb && \
# sudo apt install ./cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb && \
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub && \
# sudo apt-get update && \
# sudo apt-get -y install cuda
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
CMD gunicorn api.wsgi:application --bind 0.0.0.0:$PORT