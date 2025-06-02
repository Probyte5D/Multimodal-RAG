FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

EXPOSE 8501

CMD ["/wait-for-it.sh", "-t", "60", "milvus-standalone:19530", "--", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


