FROM python:3.7

WORKDIR /app

RUN pip install tensorflow==2.11.0 uvicorn==0.16 fastapi==0.70

COPY ./ ./

CMD ["python", "api.py"]