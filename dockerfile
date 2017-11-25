FROM ubuntu:16.04

MAINTAINER Sam Hardy

#all dependencies
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

RUN pip3 install --upgrade pip \
&& pip3 install scikit-learn \
&& pip3 install numpy \
&& pip3 install nltk \
&& pip3 install pandas \
&& pip3 install requests

RUN apt-get install python3-scipy -y

#nltk specific dependencies
RUN python3 -m nltk.downloader all

#configure application directory
ADD /app /app
WORKDIR /app

CMD python3 main.py