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
import sys
import logging
import tempfile
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

import tvm
from tvm import relay
from tvm import relax
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import generate_input_data
from tvm.target.target import Target
from tvm.runtime.vm import Executable
from tvm.runtime.module import Module

from tvm.relax.testing.relay_translator import from_relay


# benchmark utils
sys.path.append(os.path.dirname(sys.path[0]))
from utils import benchmark_configs as configs
from utils.get_network import get_relay_network

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

def ms_tune(mod, target, params):
    mod = relax.transform.BindParams("main", params)(mod)
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.relax_integration.tune_relax(
            mod=mod,
            target=target,
            params=params,
            task_scheduler="round-robin",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=1024,
            work_dir=work_dir,
        )
        ex = ms.relax_integration.compile_relax(
            db,
            mod=mod,
            target=target,
            params=params,
        )
    dev = tvm.device(str(target), 0)
    vm = relax.VirtualMachine(ex, dev)
    return ex, vm


def build(
    mod: tvm.IRModule,
    params,
    input_shape: Tuple[int],
    target: Target,
):
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod,
                    target=target,
                    params=params,
                    )
        dev = tvm.device(str(target), 0)
        vm = relax.VirtualMachine(ex, dev)
    return ex, vm

def main(_):
    compiled_suffix = configs.compiled_suffix
    batch_size = configs.batch_size
    tuning_log_path = configs.tuning_logs_path
    operator_libs_path = configs.operator_libs_path

    assert FLAGS.t in configs.available_targets, f"Unknown target: {FLAGS.t}"
    print(f"build mode: {FLAGS.mode}")
    print(f"using target: {FLAGS.t}")
    target = configs.available_targets[FLAGS.t]

    if bool(FLAGS.d):
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
        
        # to relax (from test_relay_translator.py)
        relax_mod = from_relay(mod['main'], relay_params=params, target=target)

        # Compile the optimised tensor code for each tensor operator
        if FLAGS.mode == 'tune':
            ex, vm = ms_tune(
                mod=relax_mod,
                params=params,
                input_shape=input_shape,
                target=target,
            )
        elif FLAGS.mode == 'build':
            ex, vm = build(
                mod=relax_mod,
                params=params,
                input_shape=input_shape,
                target=target,
            )
        else:
            raise RuntimeError(f"Unknown mode: {FLAGS.mode}")

        # generate input
        dev = tvm.device(str(target), 0)
        data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32), dev)

        # INFERENCE
        # res = vm["main"](data)
        # out = res.numpy()

        # BENCHMARK
        # cost = result.mean * 1e3
        res = vm.time_evaluator(func_name='main',
        dev = dev,
        number = 10,
        repeat = 1,
        )(data)
        print(f"result: ")
        print(res)


if __name__ == "__main__":
    app.run(main)
