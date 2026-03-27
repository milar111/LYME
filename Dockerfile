# Use official Python lightweight image
FROM python:3.12-slim

# Install system dependencies
# libgl1 and libglib2.0-0 are required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the web server port
EXPOSE 4000

# Run the app
CMD ["python", "main.py"]
