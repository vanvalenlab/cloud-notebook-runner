# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cloud notebook runner class for running parameterized Jupyter notebook on
google compute engine """

import os
import time
import random
import string
import googleapiclient.discovery


def randomString(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


class CloudNotebookRunner(object):
    def __init__(self,
                 input_notebook_path,
                 output_notebook_path,
                 parameters,
                 max_accelerators=4,
                 model_folder="",
                 model_bucket="",
                 data_folder="",
                 docker_image="",
                 machine_type="n1-highmem-16",
                 accelerator_type="nvidia-tesla-t4",
                 accelerators_per_node=1,
                 preemptible=False,
                 image="",
                 project="",
                 region="us-west1",
                 zone="us-west1-a",
                 date="05312020"):
        # Instance properties
        self.machine_type = machine_type
        self.accelerator_type = accelerator_type
        self.accelerators_per_node = accelerators_per_node
        self.preemptible = preemptible

        # Notebook properties
        self.input_notebook_path = input_notebook_path
        self.output_notebook_path = output_notebook_path

        # Data properties
        self.model_folder = model_folder
        self.model_bucket = model_bucket
        self.data_folder = data_folder
        self.docker_image = docker_image

        # Training parameters
        self.parameters = parameters

        # GCE properties
        self.image = image
        self.project = project
        self.region = region
        self.zone = zone
        self.data = date
        self.max_accelerators = max_accelerators

        # Create compute object
        self.compute = googleapiclient.discovery.build('compute', 'v1')

    def _poll_accelerators(self):
        quotas = self.compute.regions().get(project=self.project,
                                            region=self.region).execute()
        metric = ""
        if self.preemptible:
            metric += "PREMPTIBLE_"
        metric += "NVIDIA_"
        if "t4" in self.accelerator_type:
            metric += "T4_GPUS"
        elif "v100" in self.accelerator_type:
            metric += "V100_GPUS"
        elif "p100" in self.accelerator_type:
            metric += "P100_GPUS"

        for q in quotas['quotas']:
            if q['metric'] == metric:
                limit = q['limit']
                usage = q['usage']

        return usage, limit

    def _create_parameter_string(self, parameter):
        parameter_string = ""
        for key in parameter.keys():
            parameter_string += "-p {} {} ".format(key, str(parameter[key]))
        return parameter_string

    def _create_output_notebook_path(self, parameter):
        parameter_string = [str(key) + '_' + str(parameter[key])
                            + '_' for key in parameter]

        parameter_string = "".join(parameter_string)[0:-1] + '.ipynb'
        output_notebook_path = os.path.join(self.output_notebook_path,
                                            parameter_string)

        return output_notebook_path

    def _create_instance_name(self, parameter):
        instance_name = self.project + '-' + randomString(8)

        return instance_name

    def _create_startup_script(self, parameter):
        output_notebook_path = self._create_output_notebook_path(parameter)
        instance_name = self._create_instance_name(parameter)
        parameter_string = self._create_parameter_string(parameter)
        startup_script = """sudo docker run --gpus all -v {}:/data {} papermill {} {} {} \n
            gsutil cp -r {}/* {} \n
            sudo gcloud --quiet compute instances delete {} --zone {}""".format(self.data_folder,
                                                                              self.docker_image,
                                                                              self.input_notebook_path,
                                                                              self.output_notebook_path,
                                                                              parameter_string,
                                                                              self.model_folder,
                                                                              self.model_bucket,
                                                                              instance_name,
                                                                              self.zone)
        return startup_script

    def _create_instance(self, parameter):
        # Get instance name
        instance_name = self._create_instance_name(parameter)

        # Get the machine image link
        image_response = self.compute.images().get(project=self.project,
                                                   image=self.image).execute()
        source_disk_image = image_response['selfLink']

        # Configure the machine
        machine_type = "zones/{}/machineTypes/{}".format(self.zone,
                                                        self.machine_type)
        accelerator = "projects/{}/zones/{}/acceleratorTypes/{}".format(self.project,
                                                                        self.zone,
                                                                        self.accelerator_type)

        # Create start up script
        startup_script = self._create_startup_script(parameter)

        # Create configuration
        config = {
            # Specify name
            'name': instance_name,

            # Specify machine type
            'machineType': machine_type,

            # Specify accelerator type
            'guestAccelerators': [
                {"acceleratorType": accelerator,
                 "acceleratorCount": self.accelerators_per_node}
            ],

            # Specify whether it is pre-emptible
            'scheduling': {
                'onHostMaintenance': "TERMINATE",
                'preemptible': self.preemptible
            },

            # Specify disk
            'disks': [
                {
                    'boot': True,
                    'autoDelete': True,
                    'initializeParams': {
                        'sourceImage': source_disk_image
                    }
                }
            ],

            # Specify a network interface with NAT to access the public
            # internet.
            'networkInterfaces': [{
                'network': 'global/networks/default',
                'accessConfigs': [
                    {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                ]
            }],

            # Allow the instance to access google cloud platform.
            'serviceAccounts': [{
                'email': 'default',
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }],

            # Add metadata, which includes the startup script
            'metadata': {
                'items': [{
                    'key': 'startup-script',
                    'value': startup_script
                }]
            }
        }

        # Create the instance
        return self.compute.instances().insert(project=self.project,
                                               zone=self.zone,
                                               body=config).execute()

    def _delete_instance(self, instance_name):
        return self.compute.instances().delete(project=project,
                                               zone=zone,
                                               instance=instance_name).execute()

    def run_notebooks(self, sleep_period=60):
        number_of_notebooks = len(self.parameters)
        submitted_notebooks = 0
        while submitted_notebooks < number_of_notebooks:
            usage, limit = self._poll_accelerators()
            if usage < limit:
                parameter = parameters.pop()
                self._create_instance(parameter)
                submitted_notebooks += 1
            else:
                time.sleep(sleep_period)
