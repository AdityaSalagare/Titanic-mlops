# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY model/model.pkl /app/model/
COPY api/app.py /app/api/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
