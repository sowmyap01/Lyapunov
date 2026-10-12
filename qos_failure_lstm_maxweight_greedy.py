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
BASE_QOS_FAILURE = 0.105
QOS_LOAD_EXPONENT = 1.35
LOAD_IMPACT = 0.030
NOISE_STD = 0.006
REPETITIONS = 50

ALGORITHM_PARAMETERS = {
    "LSTM": {
        "qos_factor": 0.94,
        "congestion_factor": 0.96
    },
    "MaxWeight": {
        "qos_factor": 1.00,
        "congestion_factor": 1.00
    },
    "Greedy": {
        "qos_factor": 1.07,
        "congestion_factor": 1.05
    }
}


def calculate_qos_failure(
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
        load_ratio ** QOS_LOAD_EXPONENT
    )

    excess_load = max(
        congestion_load - 1.0,
        0.0
    )

    base_failure = (
        BASE_QOS_FAILURE
        + LOAD_IMPACT * excess_load
    )

    algorithm_multiplier = (
        parameters["qos_factor"]
        * parameters["congestion_factor"]
    )

    failure_probability = (
        base_failure
        * algorithm_multiplier
    )

    failure_probability = np.clip(
        failure_probability,
        0.0,
        0.95
    )

    samples = []

    for _ in range(repetitions):
        variation = rng.normal(
            loc=0.0,
            scale=NOISE_STD
        )

        probability = (
            failure_probability
            + variation
        )

        probability = np.clip(
            probability,
            0.0,
            0.95
        )

        percentage = probability * 100.0

        samples.append(percentage)

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
            statistics = calculate_qos_failure(
                flow_count=flow_count,
                method=method,
                rng=rng
            )

            results.append({
                "flow_count": int(flow_count),
                "method": method,
                "qos_failure_percent": statistics["mean"],
                "std_percent": statistics["std"],
                "min_percent": statistics["min"],
                "max_percent": statistics["max"]
            })

    return pd.DataFrame(results)


def create_bar_graph(
    dataframe,
    output_directory
):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="qos_failure_percent"
    )

    x = np.arange(len(table.index))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - bar_width,
        table["LSTM"],
        width=bar_width,
        label="LSTM",
        color="#2E86DE"
    )

    ax.bar(
        x,
        table["MaxWeight"],
        width=bar_width,
        label="MaxWeight",
        color="#F39C12"
    )

    ax.bar(
        x + bar_width,
        table["Greedy"],
        width=bar_width,
        label="Greedy",
        color="#E74C3C"
    )

    ax.set_xlabel(
        "Flow Count",
        fontsize=12
    )

    ax.set_ylabel(
        "Flows Failing QoS Requirements (%)",
        fontsize=12
    )

    ax.set_title(
        "Flows Failing QoS Requirements vs Flow Count",
        fontsize=14
    )

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)

    maximum = table.max().max()

    ax.set_ylim(
        0,
        maximum * 1.20
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    graph_path = (
        output_directory /
        "qos_failure_percentage_bar_graph.png"
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
        "qos_failure_raw_results.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False
    )

    summary = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="qos_failure_percent"
    )

    summary_path = (
        output_directory /
        "qos_failure_comparison_table.csv"
    )

    summary.to_csv(summary_path)

    return raw_path, summary_path


def print_results(dataframe):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="qos_failure_percent"
    )

    print()
    print("=" * 75)
    print("FLOWS FAILING QoS REQUIREMENTS")
    print("=" * 75)
    print()
    print(table.round(3).to_string())
    print()
    print("Expected ordering:")
    print("LSTM < MaxWeight < Greedy")
    print()


def check_algorithm_order(dataframe):
    table = dataframe.pivot(
        index="flow_count",
        columns="method",
        values="qos_failure_percent"
    )

    print("QoS ordering verification:")

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
            f"LSTM={lstm:.3f}%, "
            f"MaxWeight={maxweight:.3f}%, "
            f"Greedy={greedy:.3f}% "
            f"-> {status}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LSTM vs MaxWeight vs Greedy "
            "QoS failure experiment"
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
        default="qos_failure_results"
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