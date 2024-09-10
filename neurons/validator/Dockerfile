FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1

RUN apt update && \
    apt install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main

COPY . /app/

RUN echo "import warnings; warnings.simplefilter('ignore')" > /usr/local/lib/python3.10/site-packages/sitecustomize.py

# Set a default command that can(should) be overridden
CMD ["python3.10", "/app/neurons/validator/main.py"]
