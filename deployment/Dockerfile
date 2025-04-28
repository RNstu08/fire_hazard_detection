FROM python:3.12-slim

WORKDIR /app

# Copy only requirements first to cache dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app (your api code etc.)
COPY api/ ./api

COPY models/ ./models
COPY ./models /app/models


# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "api/app.py"]
