# 이게 저희 aistage 서버가 쓰는 docker 이미지래요
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

COPY . /backend
WORKDIR /backend
ENV PYTHONPATH=/backend
ENV PYTHONUNBUFFERED=1

RUN pip install -r requirements.txt\
    apt update\
    apt install mysql-server\
    service mysql start
# mysql 시작할 때 뭔가 꽤 다르다. 내 로컬에서는
# $ sudo /etc/init.d/mysql start
# 이거로 하니까 됐다. 

# 다 하고 백엔드 서버 실행
RUN python3 -m src.main