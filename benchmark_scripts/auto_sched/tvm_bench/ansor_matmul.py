# from auto_scheduler_matmul_x86.ipynb

import time
import numpy as np
import tvm
from tvm import te, auto_scheduler

# constant
target = tvm.target.Target("llvm")
N = L = M = 1024
num_measure_trials = 10
log_file = "tmp/matmul.json"


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders":
               [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


task = tvm.auto_scheduler.SearchTask(func=matmul_add,
                                     args=(N, L, M, "float32"),
                                     target=target)

# Inspect the computational graph
print()
print("Computational DAG:")
print(task.compute_dag)
init_state = task.compute_dag.get_init_state()
print(init_state)  # examine more `state` structure?
print()

tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=num_measure_trials,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

t1 = time.perf_counter()

# Run auto-tuning (search)
task.tune(tune_option)

print("TUNED")

# Apply the best schedule
sch, args = task.apply_best(log_file)

t2 = time.perf_counter()

print(f"search time: {t2-t1:.2f}s")

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
