FROM python:3.12.13-slim

RUN groupadd -r queue && useradd -r -g queue -m queue

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data && chown -R queue:queue /app

USER queue

EXPOSE 8300

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8300"]
