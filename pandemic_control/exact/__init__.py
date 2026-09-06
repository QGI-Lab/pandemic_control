"""Exact solution of the finite-horizon SEIRADHV decision problem.

The environment learns a policy; this package computes the optimal one, so that a learned
policy can be scored against it. See ``README.md`` in this directory.
"""

from .model import Setting, advance, derivative, evaluate, load_setting, reward
from .solve import as_digits, branch_and_bound, constant_policies, enumerate_all

__all__ = [
    "Setting",
    "advance",
    "as_digits",
    "branch_and_bound",
    "constant_policies",
    "derivative",
    "enumerate_all",
    "evaluate",
    "load_setting",
    "reward",
]
