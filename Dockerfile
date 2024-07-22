# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
# FROM python:3.10

#      Slim version
# FROM python:3.10-slim

#      Tensorflow version
FROM tensorflow/tensorflow:2.16.1

#      Or tensorflow to run on Apple Silicon (M1 / M2)
# FROM armswdev/tensorflow-arm-neoverse:r23.08-tf-2.13.0-eigen


# Copy everything we need into the image
COPY packagename /packagename
COPY api /api
COPY setup.py setup.py
# COPY scripts scripts
COPY requirements_docker.txt requirements_docker.txt
# COPY setup.py setup.py
# COPY credentials.json credentials.json



# Install everything
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_docker.txt
RUN pip install .


# Make directories that we need, but that are not included in the COPY
RUN mkdir /models

COPY models/yolo_crack_segm_weights_30train.pt /models/yolo_crack_segm_weights_30train.pt
COPY models/model_2.keras /models/model_2.keras
# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using

# These commands install the cv2 dependencies that are normally present on the local machine, but might be missing in your Docker container causing the issue
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# RUN python -c 'cp models/my_model.keras models/my_model.keras'

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
