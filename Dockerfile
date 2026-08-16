FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY . .

RUN uv sync --frozen

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
