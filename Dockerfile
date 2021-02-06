FROM python:3.7.9-buster

RUN apt-get update -qq && \
    apt-get install -y vim

RUN mkdir /startlens
ENV APP_ROOT /startlens
WORKDIR $APP_ROOT

COPY ./requirements.txt $APP_ROOT/requirements.txt
COPY ./requirements.lock $APP_ROOT/requirements.lock

RUN pip install -r requirements.lock

COPY . $APP_ROOT