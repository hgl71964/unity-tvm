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
import pickle
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

import tvm
from tvm import relay
from tvm.relay.transform import _ffi_api
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm import meta_schedule as ms
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
flags.DEFINE_string("mode", "default", "default/custom/print")
flags.DEFINE_integer("seed", 0, "")


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


def run_print(mod, target):
    seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True, False)
    printer = Printer()
    with target, _autotvm_silencer(), tvm.transform.PassContext(
            opt_level=4,  # run at the highest level to get all passes
            # config=pass_config,
            # disabled_pass=disabled_pass,
            instruments=[printer],
    ):
        mod = seq(mod)
    # print('Pass Stats: ')
    # for k, v in printer.relay_passes.items():
    #     print(f'{k}: {v}')
    #     print()
    return mod


def run_custom(mod, target):
    seq = get_default_passes()
    random.shuffle(seq)
    with target, _autotvm_silencer(), tvm.transform.PassContext(opt_level=4, ):
        mod = seq(mod)
    return mod


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    compiled_suffix = configs.compiled_suffix
    batch_size = configs.batch_size
    tuning_log_path = configs.tuning_logs_path
    operator_libs_path = configs.operator_libs_path

    for target_name, target in configs.available_targets.items():
        for model_name, (layout, dtype,
                         shape) in configs.available_models.items():
            print(f'trying {model_name} on {target_name}')
            try:
                mod, params, input_shape, output_shape = get_relay_network(
                    configs=configs,
                    batch_size=batch_size,
                    name=model_name,
                    layout=layout,
                    dtype=dtype,
                    workload_shape=shape)

                if FLAGS.mode == "print":
                    mod = run_print(mod, target)
                    return
                elif FLAGS.mode == "custom":
                    mod = run_custom(mod, target)
                elif FLAGS.mode == "default":
                    pass  # build will use default pass
                else:
                    raise RuntimeError(f"Unknown mode: {FLAGS.mode}")

                # build and bench
                with target, _autotvm_silencer(), tvm.transform.PassContext(
                        opt_level=4 if FLAGS.mode == "default" else 0):
                    lib: ExecutorFactoryModule = relay.build(
                        mod,
                        target=target,
                        params=params,
                    )
                    dev = tvm.device(str(target), 0)
                    graph_module = graph_executor.GraphModule(
                        lib["default"](dev))
                    result = graph_module.benchmark(dev)

            except Exception as e:
                print(f'failed {model_name} on {target_name}: {e}')
                continue

            # cost = result.mean * 1e3
            print(
                f"model: {model_name}; target: {target}, cost: {result.mean} ms"
            )

            data = {
                'model_name': model_name,
                'mean': result.mean,
                'median': result.median,
                'max': result.max,
                'min': result.min,
                'std': result.std,
                'input_shape': shape,
                'dtype': dtype,
                'layout': layout,
            }
            dir_path = f'data/{target_name}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_name = f'{model_name}_untuned_{FLAGS.mode}_{FLAGS.seed}'
            file_name += '.pkl'
            with open(f'{dir_path}/{file_name}', 'w') as f:
                pickle.dump(data, f)


if __name__ == "__main__":
    app.run(main)
