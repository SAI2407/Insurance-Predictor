
FROM python:3.11-slim


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py logging_code.py best_model.pkl ./ 
COPY Transformers ./Transformers
COPY templates ./templates


EXPOSE 8000


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


