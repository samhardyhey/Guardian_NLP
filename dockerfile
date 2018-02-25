FROM ubuntu:16.04

MAINTAINER Sam Hardy

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

#python specific dependencies
RUN pip3 install scikit-learn==0.19.1 \
&& pip3 install --upgrade scipy \
&& pip3 install numpy==1.13.3 \
&& pip3 install nltk==3.2.5 \
&& pip3 install pandas==0.20.3 \
&& pip3 install requests

#nltk specific dependencies
RUN python3.6 -m nltk.downloader all

#configure app directories
ADD /app /app
WORKDIR /app