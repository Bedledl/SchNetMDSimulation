FROM docker.io/graphcore/pytorch:3.2.0

# unfortunatley ARG is not suuported on jureca Docker build server
ENV USERNAME=riedl1
ENV USERGROUP=jusers

RUN addgroup --gid 4854 ${USERGROUP}
# TODO make add user command generally aaplicable
RUN adduser --disabled-password --gecos '' ${USERNAME} --uid 22634 --gid 4854

COPY requirements.txt /opt/requirements.txt

RUN pip3 install -r /opt/requirements.txt
