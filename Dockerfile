# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Install curl (required to download Ollama)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama locally inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Set up the application directory
WORKDIR /app

# 5. Copy your dependencies list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your Python files into the container
COPY . .

# 7. Make the startup script executable
RUN chmod +x start.sh

# 8. Set Environment Variables dynamically
# Force the app to use the lighter model and increase timeout for CPU inference
ENV DEFAULT_MODEL="qwen2.5-coder:3b"
ENV OLLAMA_BASE_URL="http://localhost:11434"
ENV OLLAMA_REQUEST_TIMEOUT_SECONDS="300"

# 9. Open Hugging Face's mandatory public port
EXPOSE 7860

# 10. Run the deployment sequence
CMD ["./start.sh"]