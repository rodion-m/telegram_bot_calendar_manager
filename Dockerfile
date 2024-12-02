# Use the official Python slim image for a lightweight base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8443

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# Add any additional dependencies your app requires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management (optional)
# If you're using Poetry instead of pip, uncomment the following lines
# RUN pip install --upgrade pip
# RUN pip install poetry
# COPY pyproject.toml poetry.lock ./
# RUN poetry install --no-dev --no-root

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# (Optional) If you have a .env file, copy it. Ensure sensitive data is handled securely.
# It's recommended to use Docker secrets or environment variables for sensitive information.
# COPY .env .

# Create a non-root user for security purposes
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Expose the port your application runs on
EXPOSE 8443

# Define the entry point for the container
# Use environment variable PORT with a default of 8443
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8443"]