FROM docker.io/graphcore/pytorch:3.2.0

# unfortunatley ARG is not suuported on jureca Docker build server
ENV USERNAME=riedl1
ENV USERGROUP=jusers

RUN addgroup --gid 4854 ${USERGROUP}
# TODO make add user command generally aaplicable
RUN adduser --disabled-password --gecos '' ${USERNAME} --uid 22634 --gid 4854

ENV WORKDIR=/home/${USERNAME}
# workdir creates directory, if it doesn't already exist
WORKDIR $WORKDIR

RUN apt-get update && apt-get install python3.8-venv -y

RUN python3 -m venv mdSimEnv
RUN . mdSimEnv/bin/activate

# copy code
# this should propably be at the end of the dockerfile,
# because the code will be modified regullary
COPY src /opt/src
COPY requirements.txt /opt/requirements.txt

RUN cp -r /opt/src ${WORKDIR}/src  \
    && cp /opt/requirements.txt ${WORKDIR}/requirements.txt

RUN pip3 install -r ${WORKDIR}/requirements.txt

#RUN chown ${USERNAME}:jusers $WORKDIR
#RUN chmod u+xrw -R $WORKDIR
