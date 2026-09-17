FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .
COPY api/ api/
COPY src/ src/
COPY schemas/ schemas/
COPY data/ data/

RUN pip install ".[api]"

RUN python src/model_trainer.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]