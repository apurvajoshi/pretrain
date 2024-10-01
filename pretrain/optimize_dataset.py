import ray 
import math
from pathlib import Path
import numpy as np
from typing import Any, Dict
from jsonargparse import CLI

# Example usage:
# input_dir = "/home/ray/efs/team/datasets/fineweb/Tiny/sample_fine_web_10BT"
# tokenizer_dir = '/home/ray/efs/cluster/checkpoints/meta-llama/Meta-Llama-3-8B'
# output_dir = '/home/ray/efs/cluster/test-runs'

SPLIT_RATIO = 0.7
CHUNK_SIZE = 2049 * 8012 # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
NUM_CPUS_PER_NODE = 90

def optimize(input_dir: str, tokenizer_dir: str, output_dir: str):
    """Optimizes and tokenzies the parquet files as streaming dataset

    Args:
        input_dir: Input path to the directory containing parquet files
        tokenizer_dir: Path to the tokenzier that will be used to token the dataset
        output_dir: Output path to the directory where the streaming dataset files are stored
    """
    initialize_ray()
    num_nodes = len(ray.nodes()) - 1 # get all nodes except head node
    inputs = [str(file) for file in Path(f"{input_dir}").rglob("*.parquet")]
    inputs.sort()
    print(f"input files: {inputs}")

    split_index = math.ceil(SPLIT_RATIO * len(inputs))
    train, val = inputs[:split_index], inputs[split_index:]

    print(f"training files: {train}")
    print(f"validation files: {val}")

    num_cpus_per_worker = ray.available_resources()['CPU'] // len(ray.nodes())

    for prefix, data in {'train':train, 'val': val}.items():
        if len(data) > 0:
            batch_input_path = [ data[x[0]:x[-1]+1] for x in np.array_split(range(len(data)), min(num_nodes, len(data)))]
            futures = [parse_filename.options(num_cpus=num_cpus_per_worker).remote(node_number, input_path, tokenizer_dir, output_dir, prefix) for node_number, input_path in enumerate(batch_input_path)]
            print(ray.get(futures))

def initialize_ray():
    # Specify the dependencies needed on each node in the cluster
    # Specify the working directory to copy the code from current directory to each node on the cluster
    ray.init(
        runtime_env={
            "pip": [
                "litgpt==0.4.1",
                "litdata==0.2.12",
                "pyarrow==16.1.0",
                "tokenizers==0.19.1",
                "pyopenssl>=24.0.0",
                "sentencepiece==0.2.0",
            ],
            "working_dir":"./",
        }
    )


# Use all the CPUs available to the node.
# This function runs an optimize worker per CPU and processes multiple parquet files in parallel.
@ray.remote(num_cpus=NUM_CPUS_PER_NODE)
def parse_filename(node_number, inputs, tokenizer_dir, output_dir, prefix='train'):
    import pyarrow.parquet as pq
    from litdata import optimize
    from functools import partial
    from litgpt.tokenizer import Tokenizer
    import os
    output_path = output_dir + "/" + prefix + "/" + str(node_number)
    print(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def tokenize_fn(filepath, tokenizer=None):
        parquet_file = pq.ParquetFile(filepath)
        # Process per batch to reduce RAM usage
        for batch in parquet_file.iter_batches(columns=["text"]):
            for text in batch.to_pandas()["text"]:
                yield tokenizer.encode(text, bos=True, eos=False)
                
    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=Tokenizer(tokenizer_dir)), # Note: Use HF tokenizer or any others
        inputs=inputs,
        output_dir=output_path,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    CLI(optimize, as_positional=False)