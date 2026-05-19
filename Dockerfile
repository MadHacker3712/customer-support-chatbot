# Best practice: build from a slim base to keep the image small
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first — Docker caches this layer.
# If only app.py changes, pip install does NOT re-run.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy fine-tuned model (run finetune.py locally first)
COPY models/chatbot-finetuned ./models/chatbot-finetuned

# Copy application code
COPY app.py .

EXPOSE 8000

ENV MODEL_DIR=./models/chatbot-finetuned
ENV MAX_NEW_TOKENS=80

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
