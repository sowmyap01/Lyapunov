#!/usr/bin/env python3

import argparse
import os
import random
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

METHODS = [
    "LSTM without Lyapunov",
    "LSTM + Lyapunov + A2C",
    "RQSMR-style",
]
FLOW_COUNTS = [100, 150, 200, 250, 300]
ACTIONS = np.array([0.25, 0.50, 0.75], dtype=np.float32)
DEVICE = torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_trace(flow_count, slots, seed):
    rng = np.random.default_rng(seed)
    base = flow_count * 0.035
    burst = np.zeros(slots)

    for t in range(slots):
        if rng.random() < 0.035:
            burst[t] = rng.uniform(1.4, 2.5)

    common = rng.lognormal(mean=0.0, sigma=0.12, size=slots)
    q1 = rng.poisson(
        base * 0.55 * common * (1 + burst)
    ).astype(float)
    q2 = rng.poisson(
        base * 0.45 * common * (1 + 0.7 * burst)
    ).astype(float)

    link = np.clip(rng.normal(1.0, 0.12, slots), 0.55, 1.25)
    mobility = np.clip(rng.beta(5, 2, slots), 0.05, 0.99)

    return {
        "arr1": q1,
        "arr2": q2,
        "link": link,
        "mobility": mobility,
    }


class ArrivalLSTM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        y, _ = self.lstm(x)
        return torch.relu(self.head(y[:, -1, :]))


class OnlineForecaster:
    def __init__(self, history=12, train_every=8):
        self.history = history
        self.train_every = train_every
        self.model = ArrivalLSTM().to(DEVICE)
        self.opt = optim.Adam(self.model.parameters(), lr=0.004)
        self.x = deque(maxlen=history)
        self.pairs = deque(maxlen=180)
        self.pending = None
        self.steps = 0
        self.last = np.array([1.0, 1.0])

    def observe_and_predict(self, current):
        current = np.asarray(current, dtype=np.float32)

        if self.pending is not None:
            self.pairs.append((self.pending.copy(), current.copy()))

        self.x.append(current.copy())
        self.steps += 1

        if len(self.x) < self.history:
            pred = current.copy()
        else:
            seq = np.asarray(self.x, dtype=np.float32)
            self.model.eval()

            with torch.no_grad():
                pred = self.model(
                    torch.tensor(seq[None, :, :])
                ).numpy()[0]

            pred = np.maximum(pred, 0.05)

        self.pending = (
            np.asarray(self.x, dtype=np.float32).copy()
            if len(self.x) == self.history
            else None
        )

        if len(self.pairs) >= 16 and self.steps % self.train_every == 0:
            batch = list(self.pairs)[-64:]

            xx = torch.tensor(
                np.stack([p[0] for p in batch]),
                dtype=torch.float32,
            )
            yy = torch.tensor(
                np.stack([p[1] for p in batch]),
                dtype=torch.float32,
            )

            self.model.train()
            self.opt.zero_grad()

            loss = nn.functional.smooth_l1_loss(
                self.model(xx), yy
            )
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )
            self.opt.step()

        self.last = pred
        return pred.copy()


class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, hidden=48, n_actions=3):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s):
        z = self.body(s)
        return self.pi(z), self.v(z).squeeze(-1)


def state_vector(q1, q2, pred, link, mobility, cap):
    return np.array([
        q1 / 120.0,
        q2 / 120.0,
        pred[0] / 12.0,
        pred[1] / 12.0,
        link,
        mobility,
        (q1 + q2) / 240.0,
        cap / 10.0,
    ], dtype=np.float32)


def lyapunov_action_filter(
    logits, q1, q2, pred, link, cap, V=0.35
):
    costs = []

    for share1 in ACTIONS:
        s1 = cap * link * share1
        s2 = cap * link * (1.0 - share1)

        nq1 = max(0.0, q1 + pred[0] - s1)
        nq2 = max(0.0, q2 + pred[1] - s2)

        drift = (
            q1 * (pred[0] - s1)
            + q2 * (pred[1] - s2)
        )

        penalty = (
            0.7 * nq1
            + 0.45 * nq2
            + 0.02 * (nq1 * nq1 + nq2 * nq2)
        )

        costs.append(
            drift / 120.0 + V * penalty / 20.0
        )

    costs = np.asarray(costs, dtype=np.float32)
    scale = float(np.std(costs)) + 1e-5

    adjusted = (
        logits
        - 0.8 * (costs - costs.min()) / scale
    )

    return adjusted


def pick_lstm_only(q1, q2, pred, link, cap):
    urgency1 = pred[0] + 0.55 * q1 + 1.2
    urgency2 = pred[1] + 0.40 * q2 + 0.5

    share = urgency1 / (urgency1 + urgency2 + 1e-6)

    return int(np.argmin(np.abs(ACTIONS - share)))


def pick_rqsmr(q1, q2, arr1, arr2, link, mobility):
    stable_link = link * (0.65 + 0.35 * mobility)

    u1 = (
        arr1 + 0.50 * q1 + 1.0
    ) / max(stable_link, 0.2)

    u2 = (
        arr2 + 0.42 * q2 + 0.7
    ) / max(stable_link, 0.2)

    share = u1 / (u1 + u2 + 1e-6)

    return int(np.argmin(np.abs(ACTIONS - share)))


def train_proposed(flow_counts, episodes, slots, seed):
    seed_everything(seed)

    net = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.0015)
    gamma = 0.96

    for ep in range(episodes):
        fc = int(flow_counts[ep % len(flow_counts)])
        trace = make_trace(
            fc, slots, seed + 10000 + ep
        )

        q1 = q2 = 0.0
        forecaster = OnlineForecaster()

        logps, vals, rewards, ents = [], [], [], []

        for t in range(slots):
            arrivals = np.array([
                trace["arr1"][t],
                trace["arr2"][t],
            ], dtype=np.float32)

            pred = forecaster.observe_and_predict(arrivals)
            cap = 8.0

            st = state_vector(
                q1, q2, pred,
                trace["link"][t],
                trace["mobility"][t],
                cap,
            )

            st_t = torch.tensor(
                st[None, :], dtype=torch.float32
            )

            logits, value = net(st_t)

            logits = lyapunov_action_filter(
                logits.detach().numpy()[0],
                q1,
                q2,
                pred,
                trace["link"][t],
                cap,
            )

            dist = torch.distributions.Categorical(
                logits=torch.tensor(logits)
            )

            action = dist.sample()
            share1 = float(ACTIONS[int(action.item())])

            service1 = (
                cap * trace["link"][t] * share1
            )
            service2 = (
                cap * trace["link"][t] * (1.0 - share1)
            )

            before1 = q1 + arrivals[0]
            before2 = q2 + arrivals[1]

            drop1 = max(0.0, before1 - 120.0)
            drop2 = max(0.0, before2 - 120.0)

            q1 = min(
                120.0,
                max(0.0, before1)
                - min(max(0.0, before1), service1),
            )

            q2 = min(
                120.0,
                max(0.0, before2)
                - min(max(0.0, before2), service2),
            )

            delay1 = 5.0 + 3.0 * q1
            delay2 = 5.0 + 3.0 * q2

            reward = -(
                0.45 * delay1 / 100.0
                + 0.25 * delay2 / 100.0
                + 1.8 * (drop1 + drop2)
                / (arrivals.sum() + 1.0)
                + 0.5 * (delay1 > 50.0)
            )

            logps.append(dist.log_prob(action))
            vals.append(value.squeeze(0))
            rewards.append(float(reward))
            ents.append(dist.entropy())

        returns = []
        ret = 0.0

        for r in reversed(rewards):
            ret = r + gamma * ret
            returns.append(ret)

        returns.reverse()

        returns_t = torch.tensor(
            returns, dtype=torch.float32
        )
        vals_t = torch.stack(vals)

        adv = returns_t - vals_t.detach()
        adv = (
            (adv - adv.mean())
            / (adv.std(unbiased=False) + 1e-6)
        )

        policy_loss = -(
            torch.stack(logps) * adv
        ).mean()

        value_loss = 0.5 * (
            returns_t - vals_t
        ).pow(2).mean()

        entropy_bonus = torch.stack(ents).mean()

        loss = (
            policy_loss
            + value_loss
            - 0.01 * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            net.parameters(), 1.0
        )

        optimizer.step()

    return net


def simulate(method, trace, actor=None, deterministic=True):
    q1 = q2 = 0.0
    forecaster = OnlineForecaster()
    rows = []

    for t in range(len(trace["arr1"])):
        arr = np.array([
            trace["arr1"][t],
            trace["arr2"][t],
        ], dtype=np.float32)

        if method == METHODS[2]:
            pred = arr.copy()
        else:
            pred = forecaster.observe_and_predict(arr)

        link = float(trace["link"][t])
        mobility = float(trace["mobility"][t])
        cap = 8.0

        if method == METHODS[0]:
            ai = pick_lstm_only(
                q1, q2, pred, link, cap
            )

        elif method == METHODS[1]:
            st = state_vector(
                q1, q2, pred,
                link, mobility, cap,
            )

            with torch.no_grad():
                logits, _ = actor(
                    torch.tensor(
                        st[None, :],
                        dtype=torch.float32,
                    )
                )

            adjusted = lyapunov_action_filter(
                logits.numpy()[0],
                q1,
                q2,
                pred,
                link,
                cap,
            )

            if deterministic:
                ai = int(np.argmax(adjusted))
            else:
                ai = int(
                    torch.distributions.Categorical(
                        logits=torch.tensor(adjusted)
                    ).sample().item()
                )

        else:
            ai = pick_rqsmr(
                q1, q2, arr[0], arr[1],
                link, mobility,
            )

        s1 = cap * link * float(ACTIONS[ai])
        s2 = cap * link * (
            1.0 - float(ACTIONS[ai])
        )

        b1 = q1 + arr[0]
        b2 = q2 + arr[1]

        d1 = max(0.0, b1 - 120.0)
        d2 = max(0.0, b2 - 120.0)

        q1 = max(
            0.0,
            min(120.0, b1)
            - min(max(0.0, b1), s1),
        )

        q2 = max(
            0.0,
            min(120.0, b2)
            - min(max(0.0, b2), s2),
        )

        lat1 = 5.0 + 3.0 * q1
        lat2 = 5.0 + 3.0 * q2

        arrivals = float(arr.sum())
        drops = d1 + d2

        drop_rate = (
            100.0 * drops / max(arrivals, 1.0)
        )

        qos_fail = 100.0 * (
            float(lat1 > 50.0)
            + float(drops > 0.0)
        ) / 2.0

        rows.append({
            "slot": t,
            "latency_ms": 0.6 * lat1 + 0.4 * lat2,
            "qos_failure_pct": qos_fail,
            "packet_drop_pct": drop_rate,
            "q1_backlog": q1,
            "q2_backlog": q2,
            "action_q1_share": ACTIONS[ai],
        })

    return pd.DataFrame(rows)


def run(args):
    os.makedirs(args.out, exist_ok=True)

    print(
        "Training proposed controller on separate synthetic episodes..."
    )

    actor = train_proposed(
        FLOW_COUNTS,
        args.train_episodes,
        args.train_slots,
        args.seed,
    )

    actor.eval()

    all_rows = []
    trace_rows = []

    for fc in FLOW_COUNTS:
        for run_id in range(args.runs):
            trace_seed = (
                args.seed + 500000 + fc * 100 + run_id
            )

            trace = make_trace(
                fc, args.slots, trace_seed
            )

            for method in METHODS:
                df = simulate(
                    method, trace, actor=actor
                )

                for metric in [
                    "latency_ms",
                    "qos_failure_pct",
                    "packet_drop_pct",
                ]:
                    all_rows.append({
                        "flow_count": fc,
                        "run": run_id,
                        "method": method,
                        "metric": metric,
                        "value": float(df[metric].mean()),
                    })

                if method == METHODS[1] and run_id == 0:
                    tmp = df.copy()
                    tmp.insert(0, "flow_count", fc)
                    trace_rows.append(tmp)

        print(f"Finished flow count {fc}")

    raw = pd.DataFrame(all_rows)

    raw.to_csv(
        os.path.join(args.out, "raw_results.csv"),
        index=False,
    )

    summary = (
        raw.groupby([
            "flow_count",
            "method",
            "metric",
        ])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(
        os.path.join(args.out, "summary_results.csv"),
        index=False,
    )

    if trace_rows:
        pd.concat(
            trace_rows, ignore_index=True
        ).to_csv(
            os.path.join(
                args.out,
                "proposed_controller_trace.csv",
            ),
            index=False,
        )

    labels = [
        (
            "latency_ms",
            "Mean end-to-end latency proxy (ms)",
            "latency_line.png",
        ),
        (
            "qos_failure_pct",
            "QoS failure (%)",
            "qos_failure_line.png",
        ),
        (
            "packet_drop_pct",
            "Packet drop (%)",
            "packet_drop_line.png",
        ),
    ]

    for metric, ylabel, filename in labels:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        part = summary[
            summary["metric"] == metric
        ]

        for method in METHODS:
            p = part[
                part["method"] == method
            ].sort_values("flow_count")

            ax.plot(
                p["flow_count"],
                p["mean"],
                marker="o",
                linewidth=2,
                label=method,
            )

            if p["std"].notna().all():
                ax.fill_between(
                    p["flow_count"].to_numpy(),
                    (p["mean"] - p["std"]).to_numpy(),
                    (p["mean"] + p["std"]).to_numpy(),
                    alpha=0.12,
                )

        ax.set_xlabel("Number of flows")
        ax.set_ylabel(ylabel)
        ax.set_xticks(FLOW_COUNTS)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(args.out, filename),
            dpi=180,
        )
        plt.close(fig)

    print("\nMean results (lower is better):")

    for metric, _, _ in labels:
        print(f"\n{metric}")

        p = summary[
            summary["metric"] == metric
        ].pivot(
            index="flow_count",
            columns="method",
            values="mean",
        )

        print(p.round(3).to_string())

    print(
        f"\nSaved outputs in: {os.path.abspath(args.out)}"
    )

    print(
        "Interpretation: the proposed method is not "
        "guaranteed to win; report the measured result."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--slots", type=int, default=240)
    parser.add_argument(
        "--train-episodes", type=int, default=40
    )
    parser.add_argument(
        "--train-slots", type=int, default=180
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()