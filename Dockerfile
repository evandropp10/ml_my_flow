# Base Image
FROM python:3.8-slim

# Install packages
ADD requeriments.txt .
RUN python -m pip install -r requeriments.txt

# Create the models folde
RUN mkdir models_registry

COPY ml_my_flow.py /ml_my_flow.py
COPY model_api.py /model_api.py

CMD [ "python","model_api.py" ]

# Expose port
EXPOSE 5000
