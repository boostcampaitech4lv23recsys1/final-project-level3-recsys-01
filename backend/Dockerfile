FROM python:3.8.15-slim-buster

COPY . /backend
WORKDIR /backend
EXPOSE 8000

ENV PYTHONPATH=/backend
ENV PYTHONUNBUFFERED=1

RUN pip3 install -r requirements.txt

# 다 하고 백엔드 서버 실행
CMD ["python3", "-m", "src.main"]