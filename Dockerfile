FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements-lite.txt /app/requirements-lite.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements-lite.txt \
    && python -m nltk.downloader punkt_tab

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "ui/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
