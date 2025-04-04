FROM python:3.12-slim

WORKDIR /app

COPY dist/jsg_ml_api-*.whl /app/
COPY requirements.txt /app/
COPY main.py /app/

RUN pip install --upgrade pip && \
    pip install jsg_ml_api-*.whl

CMD ["python", "main.py"]