FROM python:3.12-slim

WORKDIR /app

# requirements 먼저 복사하고 설치
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 나머지 코드 전체 복사
COPY . .

CMD ["python", "main.py"]
