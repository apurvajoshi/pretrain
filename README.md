# pretrain

## Set up

### Install Poetry 

We use [poetry](https://python-poetry.org/) to manage and install dependencies. Install poetry by running the following command if you do not have it already installed.  

```
curl -sSL https://install.python-poetry.org | python3 -
```

### Install dependencies for the project 

Git clone the repository and install the dependencies 


```
git clone git@github.com:apurvajoshi/pretrain.git
```

```
cd pretrain
```

```
poetry install
```


## Cluster set up to train LLMs

The training is conducted on a distributed Ray cluster with the following settings: 


```
docker_image: "rayproject/ray:2.32.0-py311-gpu"

notebook:
  ui: "vscode"
  shm: "2G"
  instance_type: "p4de.24xlarge"
  az: 'us-east-1d'
  reservation: true

ray_cluster:
  head:
    ram: "40G"
    cpu: 1
    gpu: 1
  worker_groups:
    gpu_group:
      min_replicas: 4
      max_replicas: 4
      instance_type: "p4de.24xlarge"
      az: 'us-east-1a'
      reservation: true
```
