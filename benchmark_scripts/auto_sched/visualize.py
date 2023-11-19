from matplotlib import pyplot as plt
import os
import configs
from typing import Any, List, Dict, Tuple, Callable, get_type_hints
import numpy as np
import csv
from pathlib import Path
"""
This script simply prints out the end-to-end
measurement results, sorted by model -> target -> trials

This is only meant for a quick glimpse into all the existing
results. For more information, look at the README file for the
repo.
"""

Record = Dict[str, str | int]

# Plot settings
plot_width_pixels = 3840
plot_height_pixels = 2160
plot_dpi = 300

measurement_files = os.listdir(configs.MEASUREMENTS_PATH)
records: List[Record] = [
    {
        "name": file,
        "model": l[0],
        "target": l[1],
        "autoscheduled": l[2] == "ansor",
        "trials": int(l[3].split("=")[1]),
        "batchsize": int(l[4].split("=")[1]) if "batchsize" in l[4] else 1,
        "timestamp": l[-2],
        "type": l[-1].split(".")[0]
    }
    # Note the "walrus" := operator is only supported on python >= 3.8
    for file in measurement_files if len(l := file.split("_")) >= 6
]


def fn_decltype_(f: Callable):
    """ Helper function acting like decltype in C++
    returns the type hint of the return types of callables"""
    assert callable(f)
    try:
        return get_type_hints(f).get('return', Any)
    except TypeError:
        return Any

def select_by_key_combo(*keys, records: List[Record], log_type: str = "benchmarks")\
    -> Dict[Tuple[str, ...], List[Record]]:
    res: fn_decltype_(select_by_key_combo) = {}
    # res: Dict[Tuple[str, ...], List[Record]] = {}
    for record in filter(lambda r: r["type"] == log_type, records):
        key_combo: Tuple[str, ...] = tuple(record[key] for key in keys)
        res.setdefault(key_combo, [])\
           .append(record) # Get or set default and append current
    return res


def print_e2e_results_summary():
    filtered : fn_decltype_(select_by_key_combo) = \
        select_by_key_combo("model","target", "batchsize",
                      records=records, log_type="benchmarks")

    for (model, target, batchsize), rs in sorted(filtered.items()):
        batchsize = batchsize if batchsize >= 1 else 1
        print(
            f"target: { target }; model: { model }; batchsize: { batchsize }")
        for r in sorted(
                rs,
                key=lambda r:
            (r["target"], r['batchsize'], r["trials"], int(r["timestamp"]))):
            batchsize = r['batchsize'] if r['batchsize'] >= 1 else 1
            np_measurements = np.load(
                configs.MEASUREMENTS_PATH.joinpath(r["name"]))
            print(f"timestamp: {r['timestamp']} ⟐ "
                  f"trials: {r['trials']} ⟐ "
                  f"mean e2e latency (unit:ms): {np.mean(np_measurements)} ⟐ "
                  f"throughput (unit:s^{{-1}}): "
                  f"{batchsize * 1000 / np.mean(np_measurements)}")
        print("\n\n\n")


def get_plot_dict() -> Dict[Tuple[str, ...], Dict[int, np.ndarray]]:
    filtered : fn_decltype_(select_by_key_combo) = \
        select_by_key_combo("model", "target", "batchsize",
                      records=records, log_type="benchmarks")
    res: Dict[Tuple[str, ...], Dict[int, np.ndarray]] = {}
    concat_log: Dict[Tuple[int, str, ...], List[Path]] = {}
    for k, rs in sorted(filtered.items()):
        model, target, batchsize = k
        print(f"{model}-{target}-{batchsize}")
        res[k] = {}  # Init dict
        for r in sorted(rs, key=lambda r: (r["trials"], int(r["timestamp"]))):
            trials: int = r['trials']
            file_path = configs.MEASUREMENTS_PATH.joinpath(r["name"])
            if res[k].get(trials) is None:
                res[k][trials] = np.load(file_path, "r+")
                concat_log.setdefault((trials, ) + k, []).append(file_path)
            else:
                # cur_array = res[k][trials]
                new_array = np.load(file_path, "r+")
                res[k][trials] = new_array
                concat_log[(trials, ) + k].append(file_path)
    for k, v in concat_log.items():
        if len(v) > 1:
            print(f"concatenated {list(map(lambda x: x.name, v))}")
    return res


def plot_e2e_latency_graph(key: Tuple[str, ...],
                           database: Dict[Tuple[str, ...], Dict[int,
                                                                np.ndarray]]):
    x_vals = []
    means = []
    medians = []
    errors = []
    q19s = []

    model, target, batchsize = key
    for k, v in (database[key]).items():
        x_vals.append(k)
        mean = np.mean(v)
        medians.append(np.median(v))
        means.append(mean)
        max_val = np.max(v)
        min_val = np.min(v)
        # q1 = np.percentile(v, 10)
        # q9 = np.percentile(v, 90)
        # q19 = [q1, q9]
        # q19s.append(q19)
        error = np.array([[mean - min_val], [max_val - mean]])
        errors.append(error)

    x_vals = np.array(x_vals)
    vals = list(database[key].values())
    # means = np.array(means)
    # errors = np.array(errors).squeeze().T
    # q19s = np.array(q19s).squeeze().T

    plt.figure(figsize=(plot_width_pixels / plot_dpi,
                        plot_height_pixels / plot_dpi),
               dpi=plot_dpi)
    plt.boxplot(vals,
                positions=x_vals,
                widths=200,
                showmeans=True,
                meanline=True,
                showfliers=False,
                patch_artist=True,
                boxprops={
                    'facecolor': 'orange',
                    'alpha': 0.9
                },
                capprops=dict(color="black", linewidth=2))

    plt.xlim(left=0)
    plt.xticks(rotation=45)
    plt.xlabel('#Meansurement Trials')
    plt.ylabel('e2e latency (ms)')
    plt.legend()
    plt.title(
        f'e2e latency over #measure trials for {model} on {target}, batchsize={batchsize}'
    )
    plt.grid(True)

    # plt.show()
    figtitle = f"{model}-{target}-{batchsize}.png"
    plt.savefig(configs.MEASUREMENTS_PATH.joinpath(
        f"graphs/latency_v_trials/{figtitle}"),
                dpi=plot_dpi)
    print("Saved image")
    plt.close()
    print("Image Closed")


def plot_hardware_trace(workload_info: Tuple[str, str, int, int],
                        csv_file: str | os.PathLike | Path,
                        y_row_names: List[str],
                        x_row_name: List[str],
                        save_path: str | os.PathLike | Path = None):
    """
    Plots two rows from a CSV file by their row names (labels)
    on a graph and optionally saves the graph as an image.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - row_name1 (str): Label of the first row to plot.
    - row_name2 (str): Label of the second row to plot.
    - save_path (str, optional): Path to save the graph as an image. Default is None (no saving).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    model, target, batchsize, trials = workload_info

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the specified row names exist in the DataFrame's columns
    if x_row_name not in df.columns \
        or any(y_row not in df.columns for y_row in y_row_names):
        print("Invalid row names. Please provide valid column labels.")
        return

    # Get the data for the selected rows by row name
    x_data = df[x_row_name]
    y_datas = [(df[y_row_name], y_row_name)
               for y_row_name in y_row_names]  # I know, data is already plural

    # Create a new figure and plot the data
    plt.figure(figsize=(plot_width_pixels / plot_dpi,
                        plot_height_pixels / plot_dpi),
               dpi=plot_dpi)
    for y_data, y_row_name in y_datas:
        plt.plot(x_data, y_data, marker='x', linestyle='-', label=y_row_name)
    plt.xlabel('Time (s)')
    plt.ylabel("Percentage (%)")
    plt.title(
        "Hardware Utilization Percentage"
        f"for {model} on {target}, batchsize={batchsize}, trials={trials}")
    plt.legend()

    # Save the graph as an image (if save_path is provided)
    if save_path:
        plt.savefig(save_path, dpi=plot_dpi, bbox_inches='tight')

    print(f"Saved image to {save_path}")
    # Display the graph
    # plt.show()
    plt.close()
    print("Image Closed")


def main():
    # database = get_plot_dict()
    # for key, data in database.items():
    #     if len(data) > 1:
    #         print(key)
    #         plot_e2e_latency_graph(key, database)

    for (model, target, batchsize, trials), rs in \
        select_by_key_combo("model", "target", "batchsize", "trials",
                            records=records, log_type="profile").items():
        key = (model, target, batchsize, trials)
        if target == "llvm":
            continue  # Skip CPU results
        r = max(rs, key=lambda r: r['timestamp'])
        csv_path = configs.MEASUREMENTS_PATH.joinpath(r["name"])
        filename = f"{model}-{target}-{batchsize}-{r['trials']}.png"
        plot_hardware_trace(
            workload_info=key,
            csv_file=csv_path,
            y_row_names=[
                "benchmark/cuda:0 (gpu:0)/memory_utilization (%)/mean",
                "benchmark/host/cpu_percent (%)/mean",
                "benchmark/cuda:0 (gpu:0)/gpu_utilization (%)/mean",
                "benchmark/host/memory_percent (%)/mean",
                # "benchmark/cuda:0 (gpu:0)/temperature (C)/mean"
            ],
            x_row_name="benchmark/duration (s)",
            save_path=configs.MEASUREMENTS_PATH.joinpath(
                f"graphs/hardware_measurements/{filename}"))


if __name__ == "__main__":
    main()
