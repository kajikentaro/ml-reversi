FROM python:3.6

WORKDIR /app

COPY ./ ./
RUN pip install tensorflow==2.6.2 uvicorn==0.16 fastapi==0.70

CMD ["python", "api.py"]