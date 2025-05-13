FROM python:3.12-slim

WORKDIR /app

# 사전에 Makefile에서 clean이 실행된다고 가정
COPY dist/jsg_ml_api-*.whl /app/
COPY requirements.txt /app/
COPY main.py /app/

RUN pip install --upgrade pip && \
    pip install /app/jsg_ml_api-*.whl

CMD ["python", "main.py"]
