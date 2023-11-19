import numpy as np
import pandas as pd
import pickle

import psutil

from pathlib import Path

import os
import time

import tvm, tvm.relay.testing
from tvm.contrib import graph_executor

import configs
from get_network import get_relay_network

# from nvitop import ResourceMetricCollector, Device, take_snapshots
#
# from typing import Dict, Tuple, List
#
# def get_random_inputs(model_name: str, dtype : str, shape : Tuple[int, ...] ) \
#         -> Dict[str, tvm.nd.NDArray]:
#     """ Gets random input values to run the model end-to-end
#     Note if custom_get_random_inputs for the specific model is defined,
#     that value will be used instead
#
#     Parameters
#     ----------
#     model_name: str
#         The name of the model being tested
#     dtype: str
#         Data type of the input to the model
#     shape: Tuple[int, ...]
#         Shape of the input to the model, the first
#         element is always the batch size
#
#     Returns
#     -------
#     input_data: Dict[str, tvm.nd.NDArray]
#         Random input data that conforms to the input specifications of the model,
#         with each input field marked if neccesary
#     """
#     # If a custom random input is defined in configs.py, override existing
#     model_type = model_name.split("-")[0]
#     if configs.custom_get_random_inputs.get(model_name):
#         return configs.custom_get_random_inputs[model_name]()
#
#     match model_type:
#         case "bertsquad":
#             assert(type(shape) is tuple)
#             input_id_data = tvm.nd.array((np.random.uniform(size=shape,
#                                                             low=1, high=1000)).astype(dtype))
#             segment_id_data = tvm.nd.array((np.random.uniform(size=shape,
#                                                             low = 1, high = 1000)).astype(dtype))
#             input_mask_data = tvm.nd.array((np.random.uniform(size=shape,
#                                                             low = 0, high = 2)).astype(dtype))
#             unique_id_data = tvm.nd.array((np.random.uniform(size=(shape[0],),
#                                                             low = 1, high = 1000)).astype(dtype))
#             return {"unique_ids_raw_output___9:0" : unique_id_data,
#                     "input_ids:0": input_id_data,
#                     "segment_ids:0": segment_id_data,
#                     "input_mask:0": input_mask_data}
#         case "resnet" | "mobilenet":
#             assert(type(shape) is tuple)
#             data_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
#             return {"data": data_tvm}
#         case "bert" | "bert-mx":
#             assert(type(shape) is tuple)
#             input_ids_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
#             return {"input_ids": input_ids_tvm}
#         case "vit":
#             assert(type(shape) is tuple)
#             pixel_values_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
#             return {"pixel_values": pixel_values_tvm}
#         case _:
#             raise ValueError(f"Model type {model_name} does not have a sample input defined")
#
# def run_measurements(module: graph_executor.GraphModule,
#                      dev: tvm.runtime.Device,
#                      snapshot_device : str = "CUDA",
#                      number: int = 100,
#                      repeat: int = 100,
#                      min_repeat_ms: int = 500
#                      ) \
#     -> Tuple[np.ndarray, pd.DataFrame, List]:
#     """Run end-to-end benchmarks on the model, while monitoring relevant hardware states
#
#     For number, repeat and min_repeat_ms, refer to
#     https://tvm.apache.org/docs/reference/api/python/runtime.html#tvm.runtime.Module.time_evaluator
#     for more detail
#
#     Parameters
#     ----------
#     module: tvm.graph_executor.GraphModule
#         tvm graph module (including the inputs) (i.e. final compiled e2e model)
#         that will be benchmarked.
#     dev: tvm.runtime.Device
#         the target device to run on.
#     snapshot_device: str
#         Currently only supports the distinction between CUDA and non-CUDA. For CUDA,
#         nvitop will be used to take snapshots of the hardware states at fixed interval.
#         For non-CUDA, only CPU and memory usage will be monitored and recorded.
#     number: int
#         The number of times to run this function for taking average. We call these runs
#         as one repeat of measurement.
#     repeats: int
#         The number of times to repeat the measurement. In total, the function will be
#         invoked (1 + number x repeat) times, where the first one is warm up and will
#         be discarded. The returned result contains repeat costs, each of which is an
#         average of number costs.
#     min_repeat_ms: int
#         The minimum duration of one repeat in milliseconds. By default, one repeat
#         contains number runs. If this parameter is set, the parameters number will be
#         dynamically adjusted to meet the minimum duration requirement of one repeat.
#         i.e., When the run time of one repeat falls below this time, the number
#         parameter will be automatically increased. This is to ensure that the target
#         GPU is fully warmed up when the measurements are taken.
#
#     Returns
#     -------
#     benchmark_res: np.ndarray
#         An ndarray storing the end-to-end latency measurements of the model, containing
#         `repeats` data points (so by default 100). Each data point is the average of
#         running the model end-to-end for `number` times.
#     device_averaged_measurements: pd.DataFrame
#         A data frame containing multiple hardware measurements (CPU and GPU) when the
#         end-to-end tests are being conducted. The data is collected by nvitop
#         every 500ms, and the relevant data are averaged over the last 500ms.
#     device_snapshots:
#         Device snapshots (if snapshot_device=="CUDA", the GPU usage, frequency, temp,
#         VRAM usage etc.) taken every 500ms. The result is the instantaneous device
#         status at the time the snapshot is taken.
#     """
#     device_averaged_measurements = []
#     device_snapshots = []
#     device_profiling_flag = False
#
#     def on_collect(metrics : Dict[str, float]):
#         if not device_profiling_flag:
#             return False
#
#         device_averaged_measurements.append(metrics)
#         if snapshot_device == "CUDA":
#             device_snapshots.append(take_snapshots())
#         else:
#             device_snapshots.append({"timestamp": time.time(),
#                                    "cpu": (psutil.cpu_freq(), psutil.cpu_percent()),
#                                    "ram": psutil.Process(os.getpid()).memory_info()[0]})
#         return True
#
#     collector = ResourceMetricCollector(root_pids={1}, interval=0.5, devices=Device.cuda.all()) \
#                 .daemonize(on_collect=on_collect, tag="benchmark", interval=0.5, start=False) # log all devices every 1 second
#     # Log 100 data points, each time the model is evaluated 100 times or for at least 500ms, whichever takes longer
#     device_profiling_flag = True
#     collector.start()
#     ftimer = module.module.time_evaluator("run", dev, number=100, repeat=100, min_repeat_ms=500)
#     benchmark_res : np.ndarray = np.array(ftimer().results) * 1e3  # convert to millisecond
#     # cuda_profiling_flag = False
#     # collector.join()
#
#     device_averaged_res = pd.DataFrame(device_averaged_measurements)
#     return benchmark_res, device_averaged_res, device_snapshots
#
# def setup_benchmark_workload(lib, device: tvm.runtime.Device, model_name: str, shape, dtype: str):
#     """Packs up the operator library with the corresponding random input into a graph
#     module that can be benchmarked on tvm graph executor
#
#     Parameters
#     ----------
#     lib: tvm.runtime.Module | tvm.relay.backend.executor_factory.ExecutorFactoryModule | ...
#         A tvm module or factory_module (or other tvm internal representations that can
#         be executed by the graph executor) containing the optimised tensor code for
#         each tensor operator.
#     device: tvm.runtime.Device
#         The compile target of the module (or the tvm representations of the device on which
#         the model runs).
#     model_name: str
#         The name of the model, consistent with the definitions in configs.py. This
#         name will be used in :func:`~get_random_inputs~`.
#     shape: Tuple[int, ...]
#         The input shape of the model. This can be obtained from the :func:`~get_relay_network~`.
#         function as its 3rd return value. This value will be used ine :func:`~get_random_inputs~`.
#     shape: str
#         The data type the model accepts. This value will be used ine :func:`~get_random_inputs~`.
#     """
#     module = graph_executor.GraphModule(lib["default"](device)) # GraphModule, the final compiled object
#                                                                 # running on the graph executor
#     # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#     inputs : Dict[str, tvm.nd.NDArray] = get_random_inputs(model_name=model_name, shape=shape, dtype=dtype)
#     module.set_input(**inputs)
#     return module
#
# def main():
#     target = configs.target
#     compiled_lib_suffix = configs.compiled_suffix
#     batch_size = configs.batch_size
#
#     for model_name, (layout, dtype, shape) in configs.models.items():
#         print(f"Getting model parameters: {model_name}")
#         mod, params, input_shape, output_shape = get_relay_network(batch_size=batch_size,
#                                                              name=model_name,
#                                                              layout=layout,
#                                                              dtype=dtype,
#                                                              workload_shape=shape)
#         print(type(input_shape))
#         print(input_shape)
#
#         compiled_lib_name = f"{model_name}_{compiled_lib_suffix}"
#         compiled_lib_path = configs.operator_libs_path.joinpath(f"{compiled_lib_name}.so")
#
#         print(f"Loading compiled tensor operator library {model_name}...")
#         lib : tvm.runtime.Module = tvm.runtime.load_module(compiled_lib_path)
#         dev : tvm.runtime.Device = tvm.device(str(target), 0) # Target device to run on
#
#         workload = setup_benchmark_workload(lib=lib,device=dev,model_name=model_name,shape=input_shape,dtype=dtype)
#
#         print("Evaluate inference time cost...")
#
#         e2e_results, hardware_profiles, hardware_snapshops = run_measurements(module=workload, dev=dev)
#
#         print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(e2e_results), np.std(e2e_results)))
#         print(f"Raw measurements:\n {e2e_results}")
#
#         timestamp = int(time.time())
#
#         e2e_results_save = configs.MEASUREMENTS_PATH.joinpath(f"{compiled_lib_name}_{timestamp}_benchmarks.npy")
#         hardware_profile_save = configs.MEASUREMENTS_PATH.joinpath(f"{compiled_lib_name}_{timestamp}_profile.csv")
#         hardware_snapshots_save = configs.MEASUREMENTS_PATH.joinpath(f"{compiled_lib_name}_{timestamp}_snapshots.pickle")
#
#         with open(e2e_results_save, 'wb') as f:
#             np.save(f, e2e_results)
#         with open(hardware_snapshots_save, 'wb') as f:
#             pickle.dump(hardware_snapshops, f)
#         hardware_profiles.to_csv(hardware_profile_save, index=True)
#
# if __name__ == "__main__":
#     main()
#
