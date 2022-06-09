FROM python:3.9

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
EXPOSE 8000

CMD gunicorn api.wsgi:application --bind 0.0.0.0:$PORT