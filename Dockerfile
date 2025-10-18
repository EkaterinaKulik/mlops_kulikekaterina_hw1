FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

RUN curl -L -o models/model.cbm https://github.com/EkaterinaKulik/mlops_kulikekaterina_hw1/releases/download/model_v1/model.cbm

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]
