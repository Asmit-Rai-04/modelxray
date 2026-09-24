FROM python:3.12-slim
WORKDIR /app
COPY ../../pyproject.toml /app/pyproject.toml
COPY modelxray /app/modelxray
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic numpy pandas scikit-learn scipy statsmodels joblib python-multipart
EXPOSE 8000
CMD ["uvicorn", "modelxray.main:app", "--host", "0.0.0.0", "--port", "8000"]
