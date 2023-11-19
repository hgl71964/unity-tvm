import tvm
import numpy as np
import pickle
import os

from time import time

from run_compiled import setup_benchmark_workload, run_measurements
from compile_tuned import compile_with_tuning_records
from get_network import get_relay_network

import configs
import tvm.relay.testing
import tvm.contrib.graph_executor
import tvm.auto_scheduler


def main():
    target = configs.target
    target_name = configs.target_name
    compiled_lib_suffix = configs.compiled_suffix
    batch_size = configs.batch_size
    operator_libs_path = configs.operator_libs_path

    for model_name, (layout, dtype, shape) in configs.models.items():
        print(f"Getting model parameters: {model_name}")
        mod, params, input_shape, output_shape = get_relay_network(
            batch_size=batch_size,
            name=model_name,
            layout=layout,
            dtype=dtype,
            workload_shape=shape)
        log_files = [
            str(configs.tuning_logs_path.joinpath(f"{model_name}.log"))
        ]
        tasks, task_weights = tvm.auto_scheduler.extract_tasks(
            mod["main"], params, configs.target)
        l_task = len(tasks)
        start = ((l_task * 64) // 250 + 1) * 250

        i = start
        while i < max(3000, start + 1000):
            compiled_suffix = f"{target_name.lower()}_ansor_trials={i}_batchsize={batch_size}"
            os.environ["TVM_NUM_THREADS"] = str(14)
            lib = compile_with_tuning_records(mod=mod,
                                              params=params,
                                              log_files=log_files,
                                              n_lines=i)

            # Export the library for future use
            # compiled_model = f"{model_name}_{compiled_suffix}.so"
            # lib.export_library(str(operator_libs_path.joinpath(f"{compiled_model}")))
            # print(f"Exported compiled library to {compiled_model}")

            print("Seting up benchmark workload")
            device = tvm.device(str(target), 0)
            workload = setup_benchmark_workload(lib=lib,
                                                device=device,
                                                shape=input_shape,
                                                dtype=dtype,
                                                model_name=model_name)

            print("Evaluate end-to-end inference time cost...")
            e2e_results, hardware_profiles, hardware_snapshops = \
                run_measurements(module=workload, dev=device)

            print("Mean inference time (std dev): %.2f ms (%.2f ms)" % \
                (np.mean(e2e_results), np.std(e2e_results)))
            print(f"Raw measurements:\n {e2e_results}")

            timestamp = int(time())

            workload_name = f"{model_name}_{compiled_suffix}"
            e2e_results_save = configs.MEASUREMENTS_PATH.\
                joinpath(f"{workload_name}_{timestamp}_benchmarks.npy")
            hardware_profile_save = configs.MEASUREMENTS_PATH.\
                joinpath(f"{workload_name}_{timestamp}_profile.csv")
            hardware_snapshots_save = configs.MEASUREMENTS_PATH.\
                joinpath(f"{workload_name}_{timestamp}_snapshots.pickle")

            with open(e2e_results_save, 'wb') as f:
                np.save(f, e2e_results)
            with open(hardware_snapshots_save, 'wb') as f:
                pickle.dump(hardware_snapshops, f)
            hardware_profiles.to_csv(hardware_profile_save, index=True)
            print(
                f"written to files: {e2e_results_save}, {hardware_profile_save}, {hardware_snapshots_save}"
            )

            i += 250  # Greater granularity for smaller trial number


if __name__ == "__main__":
    main()
