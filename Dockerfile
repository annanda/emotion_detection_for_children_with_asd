FROM python:3.10
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
RUN mkdir -p /app/emotion_detection_system/ && \
    touch /app/emotion_detection_system/__init__.py
WORKDIR /app
RUN apt-get update \
&& apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        vim
RUN pip install -r requirements.txt && \
    rm -rf /tmp/pip* /root/.cache
ADD ./ /app
RUN pip install -e . && \
    rm -rf /tmp/pip* /root/.cache
#RUN apt-get update -y \
#    && apt-get install -y \
#    locales && locale-gen en_US.UTF-8
#ENV LANG en_US.UTF-8
#ENV LANGUAGE en_US:en
#ENV LC_ALL en_US.UTF-8