FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl

# Install gcloud - useful for debugging
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list 
RUN	apt-get install -y apt-transport-https ca-certificates gnupg
RUN	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - 
RUN	apt-get update && apt-get install -y google-cloud-sdk

WORKDIR /cloud-notebook-runner

COPY requirements.txt .
COPY model_trainer.py .

ENV GOOGLE_APPLICATION_CREDENTIALS="/root/.config/gcloud/gcloud_auth.json"
RUN pip3 install -r requirements.txt && rm requirements.txt