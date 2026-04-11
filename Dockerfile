FROM python:3.9-slim

WORKDIR /app

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY opencure/ opencure/
COPY experiments/ experiments/
COPY scripts/ scripts/
COPY setup_data.sh start.sh ./

RUN chmod +x start.sh setup_data.sh

CMD ["./start.sh"]
