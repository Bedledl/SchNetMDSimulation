FROM docker.io/graphcore/pytorch-geometric:3.2.0

RUN apt-get update && apt-get install git -y
RUN apt-get update && apt-get install vim -y

COPY requirements.txt /opt/requirements.txt

RUN mkdir ~/md_workdir

RUN pip3 install -r /opt/requirements.txt
