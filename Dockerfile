FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first so they cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY server/ ./server/
COPY templates/ ./templates/

# Runtime data: routing graphs, walks/accom datasets, GPX tracks.
COPY output/walks.csv ./output/walks.csv
COPY output/accommodation.json ./output/accommodation.json
COPY output/lake_district_walking_graph.graphml ./output/
COPY output/lake_district_driving_graph.graphml ./output/
COPY output/gpx ./output/gpx

# Cloud Run injects PORT at runtime; default 8080 for local docker run.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT is expanded by the container shell.
CMD exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT}
