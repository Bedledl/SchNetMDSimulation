FROM docker.io/graphcore/pytorch-geometric:3.2.0

RUN apt-get update && apt-get install git -y
RUN apt-get update && apt-get install vim -y

COPY requirements.txt /opt/requirements.txt

RUN mkdir ~/md_workdir

RUN pip3 install -r /opt/requirements.txt

#COPY atomwise.py /opt/atomwise.py
#COPY scatter.py /opt/scatter.py
#RUN cp /opt/atomwise.py /usr/local/lib/python3.8/dist-packages/schnetpack/atomistic/atomwise.py

#RUN cp /opt/scatter.py /usr/local/lib/python3.8/dist-packages/schnetpack/nn/scatter.py