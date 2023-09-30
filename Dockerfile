FROM python:3.7-slim

WORKDIR /app

RUn pip install numpy==1.19.2 --extra-index-url https://www.piwheels.org/simple
RUN pip install tflite-runtime==2.10.0
RUN pip install uvicorn==0.22 fastapi==0.103 --extra-index-url https://www.piwheels.org/simple
RUN apt-get update && apt-get install -y libatlas-base-dev

COPY default_model.tslite default_model.tslite
COPY src/ src/

CMD ["python", "src/api_tslite.py"]
