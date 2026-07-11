FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=180 --retries=5 \
    torch --index-url https://download.pytorch.org/whl/cpu

# RUN pip install --no-cache-dir --default-timeout=180 --retries=5 \
#     torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=180 --retries=5 -r requirements.txt

CMD ["bash"]