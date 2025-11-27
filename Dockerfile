FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock .
COPY src ./src
RUN pip install --no-cache-dir uv && uv pip install --system .
COPY logs/best_model.zip ./logs/best_model.zip
EXPOSE 8000
ENV UV_NO_SYNC=1
CMD ["nifty", "server", "--model_path", "logs/best_model.zip", "--host", "0.0.0.0", "--port", "8000"]
