# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "churn.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false"]
