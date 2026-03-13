FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first.
# sentence-transformers pulls torch as a dep — if we don't pre-install
# the CPU wheel, pip defaults to the 2GB CUDA build (useless on Railway).
# CPU wheel is ~250 MB vs ~2 GB for CUDA, keeping builds well within timeout.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies.
# pip sees torch is already satisfied and skips it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
