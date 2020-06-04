# Notebook Runner
This package uses google cloud's python client API to submit parameterized Jupyter notebooks to run on self-deleting google compute engine instances. While our lab's use for this is to train deep learning models, it can be extended to other notebooks. The NotebookRunner class does the following
* Collect a list of parameters for a parameterized Jupyter notebook that can run on papermill
* Collect configuration for google cloud instances
* Collect information of which Docker image to run, which data folder to mount to /data in the image, and which folder contains saved data (like trained models and bench marks)
* Create an instance with the specified configuration
* Run the notebook with papermill with the specified Docker image with the specified parameters
* Delete the created instance

To work, this library needs you to a few things prior to installation
* You need to have a google cloud account
* You need to have your credentials downloaded as a json called gcloud_auth.json

After installation, when starting the docker image, you need to do the following
* Mount the folder with gcloud_auth.json into the /root/.config/gcloud folder - e.g.
```bash
sudo docker run -it --privileged \
	-p 8888:8888 \
	-v ${USER}/path/to/folder:/root/.config/gcloud \
	${USER}/cloud-notebook-runner:latest bash
```