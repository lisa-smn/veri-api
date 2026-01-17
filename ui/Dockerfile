FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY ui/requirements-ui.txt /app/requirements-ui.txt
RUN pip install --no-cache-dir -r requirements-ui.txt

# Copy UI files
COPY ui/ /app/ui/
COPY docs/ /app/docs/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

