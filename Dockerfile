FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml requirements.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir /data && chown appuser:appuser /data
USER appuser
ENV HR_DATABASE_PATH=/data/hrlearnium.sqlite3
EXPOSE 8000
CMD ["uvicorn", "hrlearnium.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

