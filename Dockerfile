FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables for production
ENV ENV=production
# WEBHOOK_URL, SENTRY_DSN, etc. should be set at runtime via environment variables
# ENV WEBHOOK_URL=https://your-public-webhook-url.com

# Expose the port (assume AWS will route to this port)
EXPOSE 8443

# In production mode, main.py will set the webhook and run Uvicorn server
CMD ["python", "main.py"]