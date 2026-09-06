"""Solve the SEIRADHV decision problem exactly and compare the optimum with fixed policies.

Examples
--------
Baseline configuration on a coarse grid of twelve monthly decisions::

    uv run python scripts/run_exact_solution.py --config configs/envs_tests/default-seiradhv.json \
        --days 30 --horizon 12

The decision grid of the reported runs, solved by branch and bound::

    uv run python scripts/run_exact_solution.py --config configs/envs_tests/default-seiradhv.json \
        --method bnb --max-nodes 2000000
"""

from __future__ import annotations

import argparse

from pandemic_control.exact import (
    as_digits,
    branch_and_bound,
    constant_policies,
    enumerate_all,
    evaluate,
    load_setting,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="environment configuration file"
    )
    parser.add_argument(
        "--days", type=float, default=None, help="length of a decision interval"
    )
    parser.add_argument("--horizon", type=int, default=None, help="number of decisions")
    parser.add_argument("--method", choices=("enumerate", "bnb"), default="enumerate")
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--discount", type=float, default=1.0)
    args = parser.parse_args()

    overrides = {"discount": args.discount}
    if args.days is not None:
        overrides["days"] = args.days
    setting = load_setting(args.config, **overrides)
    horizon = args.horizon if args.horizon is not None else setting.horizon

    print(f"config              {args.config}")
    print(
        f"cohort N            {setting.N:,.0f}     hospital capacity {setting.hosp_cap:,.0f}"
    )
    print(f"decision interval   {setting.days:g} days x {horizon} decisions")
    print(
        f"health weights      {tuple(setting.health_weights)}   trade-off {tuple(setting.trade_off_weights)}"
    )
    print(
        f"thresholds          tau_I = {setting.tau_I:,.0f} people   tau_H = {setting.tau_H:,.0f} beds"
    )

    if args.method == "enumerate":
        best, sequence = enumerate_all(setting, horizon)
        print(
            f"\nsearched            {len(setting.betas) ** horizon:,} action sequences"
        )
        proved = True
        nodes = None
    else:
        best, sequence, nodes, proved = branch_and_bound(
            setting, horizon, beam_width=args.beam_width, max_nodes=args.max_nodes
        )
        print(f"\nnodes explored      {nodes:,}")

    label = "OPTIMAL" if proved else "BEST FOUND (search truncated)"
    print(f"{label:<19} J = {best:9.4f}   policy = {as_digits(sequence)}")
    print("                    (0 no restriction, 1 social distancing, 2 lockdown)\n")
    for name, value in constant_policies(setting, horizon).items():
        print(f"{name:<19} J = {value:9.4f}   gap to optimum = {best - value:8.4f}")

    _, states = evaluate([0] * horizon, setting)
    peak_infected = float((states[:, 3] + states[:, 4]).max())
    print(
        f"\nunder no restriction: peak infected {peak_infected:,.0f} (threshold {setting.tau_I:,.0f}), "
        f"peak hospital {states[:, 5].max():,.0f} (threshold {setting.tau_H:,.0f}), "
        f"deceased {states[-1, 7]:,.0f}"
    )


if __name__ == "__main__":
    main()
