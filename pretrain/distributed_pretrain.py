import ray
import os
from litgpt import Config
from litgpt.args import TrainArgs, EvalArgs
from litgpt.data import TextFiles
from litgpt.data import LitData
from pathlib import Path
import torch
from pretrain import setup
import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train import CheckpointConfig
from ray.train import FailureConfig
from jsonargparse import CLI
from typing_extensions import Literal
from typing import Optional, Tuple, Union, Dict


NUM_OF_GPUS_PER_NODE = 8
LOGGING_DIR = "/home/ray/efs/cluster/" # Directory to store ray logs

def pretrain(
    model_name: str,
    out_dir: Path,
    dataset_path: Path,
    num_nodes:int = 1,
    model_config: Optional[Config] = None,
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = "bf16-true",
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=8,
        max_tokens=100_000_000,
        max_seq_length = 2048,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    tokenizer_dir: Optional[Path] = Path("/home/ray/efs/cluster/checkpoints/meta-llama/Meta-Llama-3-8B/"),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    devices: Union[int, str] = "auto",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    seed: int = 42
):
    """ 
    Distributed pretraining of a model.

    Arguments:
        dataset_path: Path to dataset. Assumes the path contains two subdirectories "train" and "val" with training and valiation dataset
        num_nodes: Number of nodes in the cluster
        out_dir: Directory in which to save checkpoints
        model_name: (Optional) The name of the model to pretrain. Choose from names in ``litgpt.config``. Set to "Meta-Llama-3-8B" by default.
        max_tokens: (Optional) Total number of tokens to train on
        micro_batch_size:(Optional) Number of samples per data-parallel rank
        max_seq_length: (Optional) Limits the length of samples
        precision: (Optional) The precision to use for pretraining.
        max_norm: (Optional) Optimization argument. By default, set to 1.0
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset.
    """
    initialize_ray()

    # [1] Define training config
    train_loop_config = {"model_name": model_name, "out_dir": out_dir, "dataset_path": dataset_path, "num_nodes": num_nodes,  "model_config": model_config,
    "precision":precision, "train": train, "tokenizer_dir": tokenizer_dir, "eval": eval, "optimizer": optimizer, "devices": devices,
    "logger_name": logger_name, "initial_checkpoint_dir":initial_checkpoint_dir, "resume":resume, "seed": seed}
    print(train_loop_config)

    # [2] Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=num_nodes * NUM_OF_GPUS_PER_NODE, use_gpu=True)

    # [3] Launch distributed training job.
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            storage_path=LOGGING_DIR,
            checkpoint_config=CheckpointConfig(checkpoint_frequency=0, num_to_keep=None),
            failure_config=FailureConfig(max_failures=0)),
    )
    result: ray.train.Result = trainer.fit()
    ray.shutdown()
 

def initialize_ray():
    print(f"WANDB Host: {os.environ['WANDB_HOST']} WANDB Key Used: {os.environ['WANDB_API_KEY']}")
    ray.init(
        runtime_env={
            "pip": [
                "litgpt==0.4.1",
                "litdata==0.2.12",
                "tokenizers==0.19.1",
                "sentencepiece==0.2.0",
                "pyopenssl>=24.0.0"
            ],
            "working_dir":"./",
            "env_vars": {
                # pass custom host and api keys as env variables
                'WANDB_API_KEY': os.environ['WANDB_API_KEY'],
                'WANDB_HOST': os.environ['WANDB_HOST']
            }
        }
    )


# define training loop per worker
def train_loop_per_worker(config):
    torch.set_float32_matmul_precision("high")
    setup(model_name=config["model_name"], num_nodes=config["num_nodes"], gpus_per_node=NUM_OF_GPUS_PER_NODE, model_config=config["model_config"],
        out_dir=Path(config["out_dir"]), precision=config["precision"], initial_checkpoint_dir=config["initial_checkpoint_dir"], resume=config["resume"],
        # data=TextFiles(train_data_path=Path("/home/ray/efs/cluster/data/custom_texts")),
        data=LitData(data_path=Path(config["dataset_path"]), split_names=("train", "val")),
        train=config["train"],
        eval=config["eval"],
        optimizer=config["optimizer"],
        devices=config["devices"],
        tokenizer_dir=Path(config["tokenizer_dir"]),
        logger_name="wandb",
        seed=config["seed"]
    )



if __name__ == "__main__":
    CLI(pretrain, as_positional=False)