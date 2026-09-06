"""Exact SEIRADHV dynamics and reward, read from the same configuration files as the environment.

The environment in ``pandemic_control.environment.seiradhv`` integrates the model with
``scipy.integrate.odeint`` one decision at a time and scores the reward *after* the integration.
This module reproduces both, without gymnasium, so that the resulting finite-horizon decision
problem can be solved exactly rather than learned.

Three conventions are taken from the environment and must stay in step with it:

* ``p_s`` is the fraction of exposed people who become **symptomatic**:
  ``dI_s/dt`` carries ``p_s * delta * E`` and ``dI_a/dt`` carries ``(1 - p_s) * delta * E``.
* the reward is **post-transition**: the health term is evaluated at the state reached at the
  end of the decision interval, and the economic term is attached to the action taken.
* the health thresholds are hard-coded in ``SEIRADHV_Env.reward``: the infection term switches
  on at ``0.10 * N`` and the hospital term at ``0.70 * hosp_cap``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _load_actions() -> dict[int, list]:
    """Read ``ACTIONS_STRUCT`` from ``pandemic_control.environment.utils``.

    The relative import is tried first. When this module is used on its own, importing the
    parent package would pull in the training stack, so the table is then read straight from
    its source file. Either way there is a single definition of the actions.
    """
    try:  # normal use, inside the installed package
        from ..environment.utils import ACTIONS_STRUCT
    except ImportError:  # standalone use, e.g. to re-check a published number
        import importlib.util

        source = os.path.join(
            os.path.dirname(__file__), os.pardir, "environment", "utils.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_pc_env_utils", os.path.abspath(source)
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ACTIONS_STRUCT = module.ACTIONS_STRUCT
    return ACTIONS_STRUCT


ACTIONS_STRUCT = _load_actions()

# Thresholds hard-coded in SEIRADHV_Env.reward.
INFECTION_FRACTION = 0.10
HOSPITAL_MARGIN = 0.70

# Durations in the configuration files that the environment inverts into rates.
_INVERTED = {"gamma", "delta", "theta", "mu", "sigma"}


@dataclass(frozen=True)
class Setting:
    """Everything needed to define one instance of the decision problem."""

    N: float
    hosp_cap: float
    health_weights: tuple[float, float, float]
    trade_off_weights: tuple[float, float]
    days: float
    horizon: int
    delta_i: float
    delta_a: float
    gamma_a: float
    gamma_s: float
    gamma_h: float
    mu_s: float
    mu_h: float
    theta: float
    sigma: float
    omega: float
    rho: float
    p_h: float
    p_d: float
    p_dh: float
    y0: tuple[float, ...]
    betas: tuple[float, ...]
    eco: tuple[float, ...]
    infection_fraction: float = INFECTION_FRACTION
    hospital_margin: float = HOSPITAL_MARGIN
    discount: float = 1.0

    @property
    def tau_I(self) -> float:
        return self.infection_fraction * self.N

    @property
    def tau_H(self) -> float:
        return self.hospital_margin * self.hosp_cap


def load_setting(cfg: dict | os.PathLike | str, **overrides) -> Setting:
    """Build a :class:`Setting` from an environment configuration file or dictionary.

    Parameters mirror ``Base_Env.__init__``: durations listed under ``spec-params`` are
    inverted into rates, ``days_per_restrict`` is the length of a decision interval, and the
    number of decisions is ``round(max_steps / days)``.
    """
    if isinstance(cfg, dict):
        config = cfg
    else:
        with open(cfg, "r") as handle:
            config = json.load(handle)

    env = config["env-params"]
    spec = {
        key: ([1 / v for v in value] if isinstance(value, list) else 1 / float(value))
        if key in _INVERTED
        else value
        for key, value in config["spec-params"].items()
    }
    init = config["init-conds"]

    p_s, p_h, p_d, p_dh = spec["probas"]
    gamma_a, gamma_s, gamma_h = spec["gamma"]
    mu_s, mu_h = spec["mu"]
    delta = spec["delta"]
    days = float(env.get("days_per_restrict", 7))

    N = float(env["N"])
    y0 = (
        N
        - sum(
            float(init.get(k, 0))
            for k in ("V0", "E0", "I_a0", "I_s0", "H0", "R0", "D0")
        ),
        float(init.get("V0", 0)),
        float(init.get("E0", 0)),
        float(init.get("I_a0", 0)),
        float(init.get("I_s0", 1)),
        float(init.get("H0", 0)),
        float(init.get("R0", 0)),
        float(init.get("D0", 0)),
    )

    setting = Setting(
        N=N,
        hosp_cap=float(env["hosp_cap"]),
        health_weights=tuple(env.get("health_weights", (0.75, 1.5, 7.0))),
        trade_off_weights=tuple(env.get("trade_off_weights", (1.0, 1.0))),
        days=days,
        horizon=round(float(env["max_steps"]) / days),
        delta_i=p_s * delta,
        delta_a=(1 - p_s) * delta,
        gamma_a=gamma_a,
        gamma_s=gamma_s,
        gamma_h=gamma_h,
        mu_s=mu_s,
        mu_h=mu_h,
        theta=spec["theta"],
        sigma=spec["sigma"],
        omega=spec["omega"],
        rho=spec["rho"],
        p_h=p_h,
        p_d=p_d,
        p_dh=p_dh,
        y0=y0,
        betas=tuple(ACTIONS_STRUCT[a][0] for a in sorted(ACTIONS_STRUCT)),
        eco=tuple(ACTIONS_STRUCT[a][1] for a in sorted(ACTIONS_STRUCT)),
    )
    return setting if not overrides else _replace(setting, **overrides)


def _replace(setting: Setting, **overrides) -> Setting:
    fields = {k: getattr(setting, k) for k in setting.__dataclass_fields__}
    fields.update(overrides)
    return Setting(**fields)


def derivative(state: np.ndarray, beta: float, s: Setting) -> np.ndarray:
    """Right-hand side of the SEIRADHV equations, vectorised over a batch of states.

    ``state`` has shape ``(n, 8)`` with columns ``S, V, E, I_a, I_s, H, R, D``. The exit rates
    of the symptomatic and hospitalised compartments carry the same input fractions as
    ``SEIRADHV_Env.deriv``.
    """
    S, V, E, I_a, I_s, H, R, _D = state.T
    prevalence = (I_a + I_s) / s.N
    new_from_S = beta * S * prevalence
    new_from_V = s.rho * V * prevalence
    exit_s = (1 - (s.p_d + s.p_h)) * s.gamma_s + s.p_d * s.mu_s + s.p_h * s.theta
    exit_h = s.p_dh * s.mu_h + (1 - s.p_dh) * s.gamma_h
    return np.stack(
        [
            s.sigma * R - new_from_S - s.omega * S,
            s.omega * S - new_from_V,
            new_from_S + new_from_V - (s.delta_i + s.delta_a) * E,
            s.delta_a * E - s.gamma_a * I_a,
            s.delta_i * E - exit_s * I_s,
            s.p_h * s.theta * I_s - exit_h * H,
            s.gamma_a * I_a
            + (1 - (s.p_d + s.p_h)) * s.gamma_s * I_s
            + (1 - s.p_dh) * s.gamma_h * H
            - s.sigma * R,
            s.p_d * s.mu_s * I_s + s.p_dh * s.mu_h * H,
        ],
        axis=1,
    )


def advance(
    state: np.ndarray, beta: float, s: Setting, days: float, step: float = 0.5
) -> np.ndarray:
    """Integrate the equations for ``days`` days with classical Runge-Kutta of order four."""
    n_steps = max(1, round(days / step))
    dt = days / n_steps
    for _ in range(n_steps):
        k1 = derivative(state, beta, s)
        k2 = derivative(state + 0.5 * dt * k1, beta, s)
        k3 = derivative(state + 0.5 * dt * k2, beta, s)
        k4 = derivative(state + dt * k3, beta, s)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def reward(state: np.ndarray, action: int, s: Setting) -> np.ndarray:
    """Reward of ``SEIRADHV_Env.reward``, evaluated at the post-transition state."""
    alpha_i, alpha_h, alpha_d = s.health_weights
    I_a, I_s, H, D = state[:, 3], state[:, 4], state[:, 5], state[:, 7]
    infected = I_a + I_s
    infection = np.where(infected < s.tau_I, 0.0, -infected / s.N)
    hospital = np.where(H < s.tau_H, 0.0, -(H - s.tau_H) / s.tau_H)
    health = alpha_i * infection + alpha_h * hospital + alpha_d * (-D / s.N)
    return s.trade_off_weights[0] * health + s.trade_off_weights[1] * s.eco[action]


def evaluate(
    actions: Sequence[int], s: Setting, step: float = 0.5
) -> tuple[float, np.ndarray]:
    """Return the discounted return of one action sequence and the states it visits."""
    state = np.array([s.y0], dtype=float)
    visited = [state[0].copy()]
    total = 0.0
    for k, action in enumerate(actions):
        state = advance(state, s.betas[action], s, s.days, step)
        total += (s.discount**k) * float(reward(state, action, s)[0])
        visited.append(state[0].copy())
    return total, np.array(visited)
