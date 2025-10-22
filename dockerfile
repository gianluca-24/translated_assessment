# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

# Copy project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8001

# Command to run the FastAPI server with uvicorn
CMD ["uvicorn", "server:app","--port", "8001"]