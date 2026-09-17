FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .
COPY ui/ ui/
COPY schemas/ schemas/

RUN pip install ".[ui]"

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]