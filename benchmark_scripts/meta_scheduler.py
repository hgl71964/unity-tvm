# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring
import os
import logging
import tempfile
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import generate_input_data
from tvm.target.target import Target
from tvm.runtime.vm import Executable
from tvm.runtime.module import Module

import tvm.contrib.graph_executor as graph_executor

# benchmark utils
import auto_sched.benchmark_configs as configs
from auto_sched.get_network import get_relay_network

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("d", 0, "debug or not")
flags.DEFINE_string("mode", "build", "compile mode")
flags.DEFINE_string("t", "llvm", "target")

# This script is modified from https://github.com/apache/tvm/blob/main/tests/python/integration/test_tuning.py
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


def ms_tune(
    mod: tvm.IRModule,
    params,
    input_shape: Tuple[int],
    target: Target,
):
    with tempfile.TemporaryDirectory() as work_dir:
        with ms.Profiler() as profiler:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                params=params,
                work_dir=work_dir,
                max_trials_global=configs.num_measure_trials,
            )
            lib: ExecutorFactoryModule = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
                backend='graph',
            )

        # must exit profiler scope
        print(f'profiler:: ')
        print(profiler.table())

        # 'llvm' -> tvm.cpu(0)
        device = tvm.device(str(target), 0)
        graph_module = graph_executor.GraphModule(lib["default"](device))
    return graph_module


def build(
    mod: tvm.IRModule,
    params,
    input_shape: Tuple[int],
    target: Target,
):
    with tvm.transform.PassContext(opt_level=3):
        lib: ExecutorFactoryModule = relay.build_module.build(mod,
                                                              target=target,
                                                              params=params)
        dev = tvm.device(str(target), 0)
        graph_module = graph_executor.GraphModule(lib["default"](dev))
    return graph_module


def main(_):
    compiled_suffix = configs.compiled_suffix
    batch_size = configs.batch_size
    tuning_log_path = configs.tuning_logs_path
    operator_libs_path = configs.operator_libs_path

    assert FLAGS.t in configs.available_targets, f"Unknown target: {FLAGS.t}"
    print(f"build mode: {FLAGS.mode}")
    print(f"using target: {FLAGS.t}")
    target = configs.available_targets[FLAGS.t]

    if FLAGS.d:
        # so that python stops and we can attach debugger
        pid = os.getpid()
        input(f"attach to pid {pid}")

    # For each model listed in config, load the relay representation of the
    # model, compiles to tvm relay factory module, and store the results into
    # individual files, whose file names are specified by the model name and
    # the suffixes defined above.
    for model_name, (layout, dtype, shape) in configs.models.items():
        mod, params, input_shape, output_shape = get_relay_network(
            configs=configs,
            batch_size=batch_size,
            name=model_name,
            layout=layout,
            dtype=dtype,
            workload_shape=shape)

        # Compile the optimised tensor code for each tensor operator
        print(f"Tuning and Compiling {model_name} with {FLAGS.mode}...")
        if FLAGS.mode == 'tune':
            graph_module = ms_tune(
                mod=mod,
                params=params,
                input_shape=input_shape,
                target=target,
            )
        elif FLAGS.mode == 'build':
            graph_module = build(
                mod=mod,
                params=params,
                input_shape=input_shape,
                target=target,
            )
        else:
            raise RuntimeError(f"Unknown mode: {FLAGS.mode}")

        # inputs = generate_input_data(
        #     input_shape=input_shape,
        #     input_dtype=dtype,
        #     low=0.,
        #     high=1.,
        # )
        dev = tvm.device(str(target), 0)
        result = graph_module.benchmark(dev)

        # cost = result.mean * 1e3
        print(f"results: ")
        print(result)

        # Export the library for future use; TODO linker fail in macbook!
        # compiled_model = f"{model_name}_{compiled_suffix}.so"
        # lib.export_library(str(operator_libs_path.joinpath(f"{compiled_model}")))
        # print(f"Exported compiled library to {compiled_model}")


if __name__ == "__main__":
    app.run(main)
