FROM python:3.7

WORKDIR /app

RUN pip install tensorflow-cpu==2.11 uvicorn==0.22 fastapi==0.103

COPY default_model/ default_model/
COPY src/ src/

CMD ["python", "src/api.py"]