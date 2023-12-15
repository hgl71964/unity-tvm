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
import random
import tempfile
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

import tvm
from tvm import relay
from tvm.relay.transform import _ffi_api
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import generate_input_data
from tvm.target.target import Target
from tvm.runtime.vm import Executable
from tvm.runtime.module import Module

import tvm.contrib.graph_executor as graph_executor
from tvm.meta_schedule.relay_integration import _autotvm_silencer

# benchmark utils
sys.path.append(os.path.dirname(sys.path[0]))
from utils import benchmark_configs as configs
from utils.get_network import get_relay_network

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("d", 0, "debug or not")
flags.DEFINE_string("t", "llvm", "target")
flags.DEFINE_integer("opt_level", 4, "")
flags.DEFINE_integer("b", 1, "whether build and bench")
flags.DEFINE_integer("seed", 0, "")

# This script is modified from https://github.com/apache/tvm/blob/main/tests/python/integration/test_tuning.py
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


class Printer(tvm.ir.instrument.PassInstrument):

    def __init__(self):
        tvm.ir.instrument.PassInstrument.__init__(self)
        self.relay_passes = {}
        self.tir_passes = {}
        self.deps = {}
        self.blacklist = set("sequential", )

    def run_before_pass(self, mod, info):
        if info.name in self.blacklist:
            return

        if info.name[:3] == "tir":
            self.tir_passes[info.name] = info.opt_level
        else:
            self.relay_passes[info.name] = info.opt_level
        self.deps[info.name] = info.required

    def run_after_pass(self, mod, info):
        pass


def SimplifyExprPostAlterOp():
    return _ffi_api.SimplifyExprPostAlterOp()


def get_default_passes(passes: dict[str, int]):
    availables = [
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.qnn.transform.Legalize(),
        relay.transform.InferType(),
        relay.qnn.transform.CanonicalizeOps(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.BackwardFoldScaleAxis(),
        relay.transform.ForwardFoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        SimplifyExprPostAlterOp(),
        relay.transform.FastMath(),
    ]
    return availables


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    compiled_suffix = configs.compiled_suffix
    batch_size = configs.batch_size
    tuning_log_path = configs.tuning_logs_path
    operator_libs_path = configs.operator_libs_path

    assert FLAGS.t in configs.available_targets, f"Unknown target: {FLAGS.t}"
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

        # run (default) passes
        copy = mod
        # print(f"before pass: {id(copy)} {id(mod)}")
        seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True,
                                                                    False)
        printer = Printer()
        with target, _autotvm_silencer(), tvm.transform.PassContext(
                opt_level=4,  # run at the highest level to get all passes
                # config=pass_config,
                # disabled_pass=disabled_pass,
                instruments=[printer],
        ):
            copy = seq(copy)
        # print(f"after pass: {id(copy)} {id(mod)}")
        # print('Pass Stats: ')
        # for k, v in printer.relay_passes.items():
        #     print(f'{k}: {v}')
        #     print()

        # build and bench
        if bool(FLAGS.b):
            with target, _autotvm_silencer(), tvm.transform.PassContext(
                    opt_level=FLAGS.opt_level,
                    # config=pass_config,  # TODO use config to specify passes
                    # disabled_pass=disabled_pass,
            ):
                lib: ExecutorFactoryModule = relay.build(
                    mod,
                    target=target,
                    params=params,
                )
                dev = tvm.device(str(target), 0)
                graph_module = graph_executor.GraphModule(lib["default"](dev))
                result = graph_module.benchmark(dev)

                # cost = result.mean * 1e3
                print(f"results: ")
                print(result)


if __name__ == "__main__":
    app.run(main)
