FROM docker.io/graphcore/pytorch:3.2.0

ARG USERNAME
# riedl1
ARG USERGROUP
# jusers

RUN addgroup --gid 4854 ${USERGROUP}
# TODO make add user command generally aaplicable
RUN adduser --disabled-password --gecos '' ${USERNAME} --uid 22634 --gid 4854

ENV WORKDIR=/home/${USERNAME}
# workdir creates directory, if it doesn't already exist
WORKDIR $WORKDIR
RUN chown ${USERNAME}:jusers $WORKDIR
RUN chmod u+xrw -R $WORKDIR

RUN python3 -m venv mdSimEnv && cd mdSimEnv
RUN bin/activate

# copy code
# this should propably be at the end of the dockerfile,
# because the code will be modified regullary
COPY src src
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
