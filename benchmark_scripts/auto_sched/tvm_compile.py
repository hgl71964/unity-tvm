import tvm
from tvm.driver import tvmc
from time import time
from os import cpu_count
import os


def print_with_padding(s: str):
    """ Print string with line padding

    Parameters:
        s (str): content to print

    Returns:
        None
    """
    columns = 80
    str_len = len(s)
    if str_len > columns:
        print(s)
    else:
        padding = "=" * ((columns - str_len - 2) // 2)
        print(f"{padding} {s} {padding}")


# Use 3/4 of total threads available; if cpu_count() is not available,
# default to assume the system has 8 threads
def main():
    threads = str((cpu_count() or 8) * 2 // 3)
    os.environ["TVM_NUM_THREADS"] = threads
    print_with_padding(f"Running resnet50-v2-7 test with {threads} threads")

    print_with_padding("Loading the Model")
    start = time()
    model = tvmc.load('./resnet50-v2-7.onnx')
    end = time()
    print(f"\n{end - start} seconds elapsed")

    # target = "llvm"
    target = "cuda -mcpu=sm_80"

    # Saving the Relay IR representation of the model
    print_with_padding("Saving the relay file")
    start = time()
    model.save("resnet50-v2-7.relay.tar")
    end = time()
    print(f"\n{end - start} seconds elapsed")

    # Untuned compilation
    print_with_padding("Compiling (untuned) model")
    start = time()
    untuned_package = tvmc.compile(
        model,
        target=target,
        package_path=f"{target}-resnet50-v2-7-untuned.tar")
    end = time()
    print(f"\n{end - start} seconds elapsed")

    # Tuning
    print_with_padding("Tuning")
    start = time()
    tuning_record = tvmc.tune(
        model,
        target=target,
        tuning_records=f"{target}-resnet50-v2-7-tuning.log")
    end = time()
    print(f"\n{end - start} seconds elapsed")

    # Compile again with the tuned version
    print_with_padding("Compiling (tuned) model")
    start = time()
    package = tvmc.compile(model,
                           target=target,
                           tuning_records=tuning_record,
                           package_path=f"{target}-resnet50-v2-7-tuned.tar")
    end = time()
    print(f"\n{end - start} seconds elapsed")

    print_with_padding("Ansor Tuning")
    start = time()
    ansor_tuning_record = tvmc.tune(
        model,
        target=target,
        tuning_records=f"{target}-resnet50-v2-7-tuning-ansor.log",
        enable_autoscheduler=True)
    end = time()
    print(f"\n{end - start} seconds elapsed")

    # Compile and run again with the tuned version
    print_with_padding("Compiling (autotuned) model")
    start = time()
    package = tvmc.compile(
        model,
        target=target,
        tuning_records=ansor_tuning_record,
        package_path=f"{target}-resnet50-v2-7-tuned-ansor.tar")
    end = time()
    print(f"\n{end - start} seconds elapsed")


if __name__ == '__main__':
    main()
