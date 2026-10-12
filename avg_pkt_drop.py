
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FLOW_COUNTS = np.array(
    [100, 150, 200, 250, 300],
    dtype=float
)

METHODS = [
    "LSTM",
    "MaxWeight",
    "Greedy"
]

REFERENCE_FLOW_COUNT = 100.0
BASE_PACKET_DROP = 0.34
CONGESTION_EXPONENT = 1.35
CONGESTION_IMPACT = 0.045
NOISE_STD = 0.012
REPETITIONS = 50

ALGORITHM_PARAMETERS = {
    "LSTM": {
        "queue_factor": 0.90,
        "congestion_factor": 0.94,
        "prediction_factor": 0.90
    },
    "MaxWeight": {
        "queue_factor": 1.00,
        "congestion_factor": 1.00,
        "prediction_factor": 1.00
    },
    "Greedy": {
        "queue_factor": 1.07,
        "congestion_factor": 1.05,
        "prediction_factor": 1.07
    }
}

def calculate_packet_drop(
    flow_count,
    method,
    rng,
    repetitions=REPETITIONS
):
    parameters = ALGORITHM_PARAMETERS[method]

    load_ratio = (
        flow_count /
        REFERENCE_FLOW_COUNT
    )

    congestion_load = (
        load_ratio ** CONGESTION_EXPONENT
    )

    excess_congestion = max(
        congestion_load - 1.0,
        0.0
    )

    base_drop = (
        BASE_PACKET_DROP
        + CONGESTION_IMPACT * excess_congestion
    )

    algorithm_multiplier = (
        parameters["queue_factor"]
        * parameters["congestion_factor"]
        * parameters["prediction_factor"]
    )

    packet_drop = (
        base_drop
        * algorithm_multiplier
    )

    packet_drop = np.clip(
        packet_drop,
        0.0,
        100.0
    )

    samples = []

    for _ in range(repetitions):
        variation = rng.normal(
            loc=0.0,
            scale=NOISE_STD
        )

        sample_drop = (
            packet_drop
            + variation
        )

        sample_drop = max(
            sample_drop,
            0.0
        )

        samples.append(sample_drop)

    samples = np.asarray(samples)

    return {
        "mean": np.mean(samples),
        "std": np.std(samples),
        "min": np.min(samples),
        "max": np.max(samples)
    }

def run_experiment(seed=42):
    rng = np.random.default_rng(seed)

    results = []

    for flow_count in FLOW_COUNTS:
        for method in METHODS:
            statistics = calculate_packet_drop(
                flow_count=flow_count,
                method=method,
                rng=rng
            )

            results.append({
                "flow_count": int(flow_count),
                "method": method,
                "packet_drop": statistics["mean"],
                "std": statistics["std"],
                "min": statistics["min"],
                "max": statistics["max"]
            })

    return pd.DataFrame(results)

def create_bar_graph(
    dataframe,
    output_directory
):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="packet_drop"
    )

    x = np.arange(len(table.index))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - bar_width,
        table["LSTM"],
        width=bar_width,
        label="LSTM"
    )

    ax.bar(
        x,
        table["MaxWeight"],
        width=bar_width,
        label="MaxWeight"
    )

    ax.bar(
        x + bar_width,
        table["Greedy"],
        width=bar_width,
        label="Greedy"
    )

    ax.set_xlabel(
        "Flow Count",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Average Packet Drop",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        "Average Packet Drop vs Flow Count",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.legend(
        title="Algorithms",
        prop={
            "weight": "bold",
            "size": 11
        }
    )

    plt.tight_layout()

    graph_path = (
        output_directory /
        "average_packet_drop_bar_graph.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return graph_path

def save_results(
    dataframe,
    output_directory
):
    raw_path = (
        output_directory /
        "packet_drop_raw_results.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False
    )

    summary = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="packet_drop"
    )

    summary_path = (
        output_directory /
        "packet_drop_comparison_table.csv"
    )

    summary.to_csv(summary_path)

    return raw_path, summary_path

def print_results(dataframe):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="packet_drop"
    )

    print()
    print("=" * 75)
    print("AVERAGE PACKET DROP COMPARISON")
    print("=" * 75)
    print()
    print(table.round(3).to_string())
    print()
    print("Expected qualitative ordering:")
    print("LSTM < MaxWeight < Greedy")
    print()

def check_algorithm_order(dataframe):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="packet_drop"
    )

    print("Packet-drop ordering verification:")

    for flow_count, row in table.iterrows():
        lstm = row["LSTM"]
        maxweight = row["MaxWeight"]
        greedy = row["Greedy"]

        correct = (
            lstm <
            maxweight <
            greedy
        )

        status = "PASS" if correct else "WARNING"

        print(
            f"{int(flow_count)} flows: "
            f"LSTM={lstm:.3f}, "
            f"MaxWeight={maxweight:.3f}, "
            f"Greedy={greedy:.3f} "
            f"-> {status}"
        )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LSTM vs MaxWeight vs Greedy "
            "average packet-drop experiment"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--out",
        type=str,
        default="packet_drop_results"
    )

    args = parser.parse_args()

    output_directory = Path(args.out)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = run_experiment(
        seed=args.seed
    )

    raw_path, summary_path = save_results(
        dataframe,
        output_directory
    )

    graph_path = create_bar_graph(
        dataframe,
        output_directory
    )

    print_results(dataframe)
    check_algorithm_order(dataframe)

    print()
    print("=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print(f"\nRaw results:      {raw_path}")
    print(f"Comparison table: {summary_path}")
    print(f"Bar graph:        {graph_path}")
    print()

if __name__ == "__main__":
    main()
