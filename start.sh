#!/bin/bash

# 1. Start Ollama in the background
echo "Starting Ollama engine..."
ollama serve &

# Wait 5 seconds to ensure the Ollama API is fully responsive
sleep 5

# 2. Pull the optimized AI model 
# (Using 3b instead of 7b ensures it runs smoothly on the free CPU tier)
echo "Pulling the qwen2.5-coder:3b model (this might take a few minutes)..."
ollama pull qwen2.5-coder:3b

# 3. Start the Streamlit Dashboard on the Hugging Face port
echo "Starting IntelliFarm Dashboard..."
exec streamlit run main.py --server.port=7860 --server.address=0.0.0.0