import tvm
from os import path
from typing import Tuple, List, Dict, Callable
from pathlib import Path

########## Options ##########

# Lists all the available_models and relevant attributes.
# Should respect the format of:
#   "<model name>": ("<input Layout>", "<data type>", "<data layout>")
available_models: Dict[str, Tuple[str, str, Tuple[int, ...]]] = {
    "bert": ("NHWC", "int64", (128, )),
    "bert-mx": ("NHWC", "float32", (128, )),
    "bertsquad-8": ("NHWC", "int64", (256, )),
    "bertsquad-10": ("NHWC", "int64", (256, )),
    "bertsquad-12": ("NHWC", "int64", (256, )),
    "dcgan": ("NHWC", "float32", (3, 64, 64)),
    "mobilenet": ("NHWC", "float32", (224, 224, 3)),
    "resnet-18": ("NHWC", "float32", (224, 224, 3)),
    "resnet-34": ("NHWC", "float32", (224, 224, 3)),
    "resnet-50": ("NHWC", "float32", (224, 224, 3)),
    "resnet-101": ("NHWC", "float32", (224, 224, 3)),
    "resnet-152": ("NHWC", "float32", (224, 224, 3)),
    "resnet3d-18": ("NCDHW", "float32", (3, 16, 112, 112)),
    "resnet3d-34": ("NCDHW", "float32", (3, 16, 112, 112)),
    "resnet3d-50": ("NCDHW", "float32", (3, 16, 112, 112)),
    "resnet3d-101": ("NCDHW", "float32", (3, 16, 112, 112)),
    "resnet3d-152": ("NCDHW", "float32", (3, 16, 112, 112)),
    "vit": ("NCHW", "float32", (3, 224, 224)),
    "vgg-11": ("NHWC", "float32", (224, 224, 3)),
    "vgg-13": ("NHWC", "float32", (224, 224, 3)),
    "vgg-16": ("NHWC", "float32", (224, 224, 3)),
    "vgg-19": ("NHWC", "float32", (224, 224, 3)),
}

# Lists all the available compile targets
# Should respect the format of
#   "<target name>": <tvm target>
# To add more targets, refer to https://tvm.apache.org/docs/reference/api/python/target.html
# Also to add more targets, check https://github.com/apache/tvm/blob/HEAD/src/target/tag.cc
# To see CUDA/TVM's tag name for the target. New Nvidia GPU targets can be added directly
# via tvm.target.Target(<CUDA TAG>)
available_targets: Dict[str, tvm.target.Target] = {
    "A100": tvm.target.Target("nvidia/nvidia-a100"),
    "a100": tvm.target.Target("nvidia/nvidia-a100"),
    "P100": tvm.target.Target("nvidia/tesla-p100"),
    "p100": tvm.target.Target("nvidia/tesla-p100"),
    "LLVM": tvm.target.Target("llvm -num-cores 6"),
    "llvm": tvm.target.Target("llvm -num-cores 6"),
    "LLVM_AVX2": tvm.target.Target("llvm -mcpu=core-avx2"),
    # "c": tvm.target.Target("c"),
    # "ccompiler": tvm.target.Target("ccompiler"),
    # "opencl": tvm.target.Target("opencl"),
}

ROOT_PATH: Path = Path(__file__).parents[1]
SCRIPTS_PATH: Path = ROOT_PATH.joinpath("scripts")
MODELS_PATH: Path = ROOT_PATH.joinpath("src_models")
RELAY_MODELS_PATH: Path = ROOT_PATH.joinpath("relay_models")
MEASUREMENTS_PATH: Path = ROOT_PATH.joinpath("measurements")
TUNING_LOGS_PATH: Path = ROOT_PATH.joinpath("tuning_shell_logs")
TARGETS_PATH: Path = ROOT_PATH.joinpath("targets")

########## Configs ##########
# Total number of candidate programs the evolutionary search in Ansor
# will generate and measure; ansor recommended 800 * len(tasks) trials
num_measure_trials: int = 3  #2000
# Continuous Tuning: allow tuning to continue based on existing tuning log
# For instance, if continuous tuning is True, num_measure_trials = 2000,
# ref_num_measure_trials = 1000, the new tuning process will only make 1000
# additional trials, and combine the new tuning logs with the existing ones
# to generate a tuning log containing 2000 trials.
continuous_tuning: bool = False
# Reference tuning log for continuous tuning/compilation
# so tuning will continue from the where the previous tuning log stopped
# The batchsize has to be the same with the target batchsize
ref_num_measure_trials: int = 15000
# How many candidate programs the evolutionary search will actually measure
# measure in each generation.
num_measure_per_round: int = 64
# early stopping, usually set in relation to num_measure_trials
# early_stopping : int | None = num_measure_trials * 8
early_stopping: int = num_measure_trials * 8
# Batch size when running the final model
batch_size: int = 1
# Names of the models that will be benchmarked
model_names: List[str] = [
    "resnet-18",
    #   "resnet-34",
    #   "resnet-50",
    #   "resnet-101",
    #   "resnet-152",
    #   "bertsquad-8",
    #   "bertsquad-10",
    #   "bertsquad-12",
    #   "bert-mx"
    #   "vit"
]
# Custom relay model generators
custom_get_relay_network: Dict[str, Callable] = {
    # Example
    # "GPT2": from xxx import vit.get_network
}
# Custom benchmark workload generators
custom_get_random_inputs: Dict[str, Callable] = {
    # Example
    # "GPT2": from xxx import vit.get_inputs
}
# Name of the compile target that will be benchmarked
# target_name: str = "A100"
target_name: str = "LLVM"

########## Dependent Config Options ##########
# This sections contains the path definitions of files
# that affects the operations of the other scripts.
# DO NOT MODIFY ANYTHING IN THIS SECTION
models: Dict[str, Tuple[str, str, Tuple[int, ...]]] = {
    key: available_models[key]
    for key in model_names
}

# target : tvm.target.Target = available_targets[target_name]
# target: tvm.target.Target = tvm.target.Target("llvm -num-cores 6")

compiled_suffix: str
if target_name in available_targets.keys():
    compiled_suffix = f"{target_name.lower()}_ansor_trials={num_measure_trials}_batchsize={batch_size}"
else:
    compiled_suffix = f"unknown-target_ansor_trials={num_measure_trials}_batchsize={batch_size}"

tuning_logs_path : Path = TARGETS_PATH \
    .joinpath(target_name, f"trials_{num_measure_trials}_batchsize_{batch_size}")

metaschedule_db_path : Path = TARGETS_PATH \
    .joinpath(target_name, "metaschedule_db")

ref_tuning_logs_path : Path = TARGETS_PATH \
    .joinpath(target_name, f"trials_{ref_num_measure_trials}_batchsize_{batch_size}")

operator_libs_path: Path = TARGETS_PATH.joinpath(target_name, f"operator_libs")
