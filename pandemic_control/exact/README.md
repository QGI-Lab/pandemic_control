# Exact solution of the SEIRADHV decision problem

The environment in `pandemic_control/environment` *learns* an intervention policy. This package
*computes the optimal one*, so that a learned policy can be scored against it.

The controlled transition is deterministic and the initial state is fixed, so a deterministic
policy over `K` decisions is one of the `3 ** K` action sequences, and a randomised policy is an
average of those. The optimal return is therefore the largest of `3 ** K` numbers. No learning,
no approximation beyond the numerical integration.

## What it reuses from the environment

The point of this package is that it cannot drift from the simulator it is meant to describe:

| Taken from | What |
|---|---|
| `configs/*.json` | every rate, threshold, weight, cohort size, initial state and decision grid |
| `environment/utils.py` | `ACTIONS_STRUCT`, so the transmission rates and economic costs are the same table |

Three conventions are copied from `SEIRADHV_Env` and are worth stating explicitly, because they
are easy to get wrong:

- **`p_s` is the symptomatic fraction.** `dI_s/dt` carries `p_s * delta * E` and `dI_a/dt` carries
  `(1 - p_s) * delta * E`. At the default `p_s = 0.8`, four exposed people in five become
  symptomatic.
- **The reward is post-transition.** `step()` integrates first, overwrites the state, and only
  then calls `reward()`, which reads the new state. So `r(s_k, a_k) = C_health(s_{k+1}) + C_eco(a_k)`.
- **The health thresholds are hard-coded in `reward()`**, not read from the configuration: the
  infection term switches on at `0.10 * N` and the hospital term at `0.70 * hosp_cap`.

## Usage

```bash
# twelve monthly decisions, solved by full enumeration of 3^12 sequences
uv run python scripts/run_exact_solution.py \
    --config configs/envs_tests/default-seiradhv.json --days 30 --horizon 12

# the decision grid of the reported runs, solved by branch and bound
uv run python scripts/run_exact_solution.py \
    --config configs/envs_tests/default-seiradhv.json --method bnb --max-nodes 2000000
```

From Python:

```python
from pandemic_control.exact import (
    load_setting,
    enumerate_all,
    constant_policies,
    as_digits,
)

setting = load_setting("configs/envs_tests/default-seiradhv.json", days=30.0)
best, sequence = enumerate_all(setting, horizon=12)
print(best, as_digits(sequence))  # -4.6923 202200000000
print(constant_policies(setting, 12))  # returns of holding one action throughout
```

`enumerate_all` is exact and practical while `3 ** K` fits in memory, roughly `K <= 13`.
`branch_and_bound` handles finer grids: every reward component is nonpositive and the deceased
count never decreases, so from stage `k`

```
remaining return  <=  - alpha_d * (D_k / N) * sum_{j=k}^{K-1} discount ** j
```

is an admissible upper bound, and branches that cannot beat the incumbent are cut. Pass
`--max-nodes` to stop early; the result is then the best sequence found rather than a proved
optimum, and the output says so.

## Scoring a trained policy

The returns above are only comparable on the same decision grid, because the cumulative-death
term is charged once per decision: the same epidemic path scores differently under `K = 12` and
`K = 52`. To score a trained policy, evaluate its return on the grid it was trained on and
compare it with the optimum computed on that same grid.

## Reproducing the numbers in the paper

The paper's Section 5.3 uses the parameter table of the manuscript, which sets `p_dh = 0.0727`.
`configs/envs_tests/default-seiradhv.json` currently sets `p_dh = 0.07`, while the population-size
and decision-window configurations use `0.0727`. The optimal policy is `202200000000` either way;
the returns differ slightly:

| `p_dh` | optimal return | never restricting |
|---|---|---|
| 0.07 (default config) | −4.6923 | −6.1969 |
| 0.0727 (paper, other configs) | −4.7444 | −6.2855 |

Worth reconciling before the results are quoted anywhere.
