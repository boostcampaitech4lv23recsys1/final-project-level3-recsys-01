FROM ubuntu:18.04

COPY . /app
WORKDIR /app
EXPOSE 3000

RUN apt update -y && \
    apt upgrade -y && \
    apt install curl -y && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash

SHELL ["/bin/bash", "-c"]

RUN source ~/.nvm/nvm.sh && \
    nvm install 16.14.0 && \
    nvm use 16.14.0 && \
    npm install --global yarn && \
    yarn

ENV PATH $PATH:/root/.nvm/versions/node/v16.14.0/bin

CMD ["yarn", "start"]