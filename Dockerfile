FROM python:3.8-slim-bullseye

# Create the directories
RUN mkdir -p app/ models/ models/2D_versatile_HE/

# Install Stardist and tensorflow and its dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

COPY config.json /models/2D_versatile_HE/config.json
COPY thresholds.json /models/2D_versatile_HE/thresholds.json
COPY weights_best.h5 /models/2D_versatile_HE/weights_best.h5

RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5

# --------------------------------------------------------------------------------------------
# Install scripts
COPY descriptor.json /app/descriptor.json
COPY run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]
