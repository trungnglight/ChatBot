# Use official Python image
FROM python:3.11.9-slim

# Install pip + Poetry
RUN pip install --upgrade pip && pip install poetry

# Set the working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies (no virtualenv)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama pull "gemma3:4b"

# Copy the rest of the codebase
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Set Streamlit to run without needing a browser and with less verbose output
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
