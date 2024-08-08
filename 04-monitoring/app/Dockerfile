FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the torch wheel manually using curl
RUN curl -O https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp39-cp39-linux_x86_64.whl

# Install the torch wheel
RUN python -m pip install torch-2.3.1+cpu-cp39-cp39-linux_x86_64.whl

RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY . .

CMD ["streamlit", "run", "app.py"]