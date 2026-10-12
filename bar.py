#!/usr/bin/env python3
"""Generate grouped bar charts from SDVN comparison CSV output."""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = ["LSTM without Lyapunov", "LSTM + Lyapunov + A2C", "RQSMR-style"]

def locate_csv(path):
    if os.path.isdir(path):
        for name in ("summary_results.csv", "raw_results.csv"):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
        candidates = [os.path.join(path, x) for x in os.listdir(path) if x.lower().endswith(".csv")]
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError("Folder must contain summary_results.csv or raw_results.csv; alternatively pass --csv with the exact file.")
    if not os.path.isfile(path):
        raise FileNotFoundError("CSV not found: " + path)
    return path

def norm_metric(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if "latency" in s or "delay" in s: return "latency_ms"
    if "qos" in s and ("fail" in s or "violation" in s): return "qos_failure_pct"
    if "drop" in s or "packet_loss" in s or "packetloss" in s: return "packet_drop_pct"
    return s

def main():
    p = argparse.ArgumentParser(description="Create grouped bar charts from CSV results.")
    p.add_argument("--csv", default="results", help="CSV file or results folder (default: results)")
    p.add_argument("--out", default="bar_graphs", help="Folder to save PNG graphs")
    p.add_argument("--error-bars", action="store_true", help="Show standard-deviation error bars when available")
    args = p.parse_args()
    csv_path = locate_csv(args.csv)
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"flows":"flow_count", "number_of_flows":"flow_count", "algorithm":"method",
               "controller":"method", "metric_name":"metric", "result":"value",
               "avg":"mean", "average":"mean", "std_dev":"std", "standard_deviation":"std"}
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df.columns})

    # Normalize long-format CSVs from the provided SDVN script.
    if {"flow_count", "method", "metric"}.issubset(df.columns) and ("value" in df.columns or "mean" in df.columns):
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            grouped = df.groupby(["flow_count","method","metric"])["value"].agg(["mean","std"]).reset_index()
        else:
            grouped = df.copy()
            if "std" not in grouped.columns: grouped["std"] = float("nan")
    elif "flow_count" in df.columns and "method" in df.columns:
        metric_cols = [c for c in df.columns if norm_metric(c) in ("latency_ms","qos_failure_pct","packet_drop_pct")]
        if not metric_cols:
            raise ValueError("Could not find metric columns. Expected latency, QoS failure, and/or packet drop.")
        long = df.melt(id_vars=["flow_count","method"], value_vars=metric_cols, var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        grouped = long.groupby(["flow_count","method","metric"])["value"].agg(["mean","std"]).reset_index()
    else:
        raise ValueError("Unsupported CSV format. Expected columns flow_count, method, metric, value (raw_results.csv) or flow_count, method, metric, mean (summary_results.csv).")

    grouped["flow_count"] = pd.to_numeric(grouped["flow_count"], errors="coerce")
    grouped["metric"] = grouped["metric"].map(norm_metric)
    grouped["mean"] = pd.to_numeric(grouped["mean"], errors="coerce")
    grouped = grouped.dropna(subset=["flow_count","mean"])
    os.makedirs(args.out, exist_ok=True)

    specs = [
        ("latency_ms", "Mean latency (ms)", "latency_bar.png"),
        ("qos_failure_pct", "QoS failure (%)", "qos_failure_bar.png"),
        ("packet_drop_pct", "Packet drop (%)", "packet_drop_bar.png"),
    ]
    made = 0
    for metric, ylabel, filename in specs:
        part = grouped[grouped["metric"] == metric].copy()
        if part.empty: continue
        part["flow_count"] = part["flow_count"].astype(int)
        flows = sorted(part["flow_count"].unique())
        available = set(part["method"].astype(str))
        methods = [m for m in METHOD_ORDER if m in available] + sorted(available - set(METHOD_ORDER))
        pivot = part.pivot_table(index="flow_count", columns="method", values="mean", aggfunc="mean").reindex(index=flows, columns=methods)
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind="bar", ax=ax, width=0.82,
                   yerr=(part.pivot_table(index="flow_count", columns="method", values="std", aggfunc="mean")
                         .reindex(index=flows, columns=methods).to_numpy()
                         if args.error_bars and "std" in part.columns else None),
                   capsize=3 if args.error_bars else 0)
        ax.set_title(ylabel + " by flow count")
        ax.set_xlabel("Number of flows")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([str(x) for x in flows], rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(title="Method")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)
        made += 1

    if not made:
        raise ValueError("No recognized latency, QoS-failure, or packet-drop metrics found in the CSV.")
    grouped.to_csv(os.path.join(args.out, "bar_graph_data.csv"), index=False)
    print("Input CSV:", csv_path)
    print("Saved graphs and processed data to:", os.path.abspath(args.out))
    print("Created", made, "bar graph(s).")

if __name__ == "__main__":
    main()
