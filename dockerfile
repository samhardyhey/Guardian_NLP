FROM ubuntu:16.04

MAINTAINER Sam Hardy

#system dependencies
RUN apt-get -y update && apt-get install -y \
apt-utils \
python3-dev \
python3-setuptools \
build-essential \
curl \
dialog \
git \
python3 \
python3-pip

#python specific dependencies
RUN pip3 install --upgrade pip \
&& pip3 install scikit-learn==0.19.1 \
&& pip3 install --upgrade scipy \
&& pip3 install numpy==1.13.3 \
&& pip3 install nltk==3.2.5 \
&& pip3 install pandas==0.20.3
RUN pip3 install requests

#nltk specific dependencies
RUN python3 -m nltk.downloader all

#configure app directories
ADD /app /app
WORKDIR /app