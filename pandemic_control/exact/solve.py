"""Exact solution of the finite-horizon SEIRADHV decision problem.

The controlled transition is deterministic and the initial state is fixed, so a deterministic
policy over ``K`` decisions is one of the ``3 ** K`` action sequences and a randomised policy
is an average of those. The optimal return is therefore the largest of ``3 ** K`` numbers, and
no learning is involved.

Two routines compute it:

``enumerate_all``
    carries every action sequence at once as one array and integrates the whole frontier in
    parallel. Exact, and practical while ``3 ** K`` fits in memory (K up to about 13).

``branch_and_bound``
    for finer grids. Every reward component is nonpositive and the deceased count never
    decreases, so from stage ``k`` with cumulative return ``G`` and deceased count ``D_k``

        remaining return <= - alpha_d * (D_k / N) * sum_{j=k}^{K-1} discount ** j,

    which is an admissible upper bound. Branches that cannot beat the incumbent are cut. The
    answer is exact; the incumbent only decides how much is pruned.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np

from .model import Setting, advance, evaluate, reward


def enumerate_all(s: Setting, horizon: int | None = None) -> tuple[float, list[int]]:
    """Return the optimal return and an optimal action sequence by full enumeration."""
    K = s.horizon if horizon is None else horizon
    n_actions = len(s.betas)
    state = np.array([s.y0], dtype=float)
    total = np.zeros(1)
    sequences = np.zeros((1, 0), dtype=np.int8)

    for k in range(K):
        states, totals, seqs = [], [], []
        for action in range(n_actions):
            nxt = advance(state, s.betas[action], s, s.days)
            states.append(nxt)
            totals.append(total + (s.discount**k) * reward(nxt, action, s))
            seqs.append(
                np.hstack([sequences, np.full((len(state), 1), action, dtype=np.int8)])
            )
        state = np.vstack(states)
        total = np.concatenate(totals)
        sequences = np.vstack(seqs)

    best = int(np.argmax(total))
    return float(total[best]), sequences[best].tolist()


def _beam(s: Setting, K: int, width: int) -> tuple[float, list[int]]:
    """Beam search, used only to seed the branch-and-bound incumbent."""
    front = [(0.0, [], np.array([s.y0], dtype=float))]
    for k in range(K):
        candidates = []
        for total, seq, state in front:
            for action in range(len(s.betas)):
                nxt = advance(state, s.betas[action], s, s.days)
                candidates.append(
                    (
                        total + (s.discount**k) * float(reward(nxt, action, s)[0]),
                        seq + [action],
                        nxt,
                    )
                )
        front = heapq.nlargest(width, candidates, key=lambda item: item[0])
    total, seq, _ = max(front, key=lambda item: item[0])
    return total, seq


def branch_and_bound(
    s: Setting,
    horizon: int | None = None,
    beam_width: int = 40,
    max_nodes: int | None = None,
) -> tuple[float, list[int], int, bool]:
    """Return ``(optimal return, sequence, nodes explored, proved_optimal)``.

    ``proved_optimal`` is ``False`` only when ``max_nodes`` stopped the search early, in which
    case the returned sequence is the best one found rather than a proved optimum.
    """
    K = s.horizon if horizon is None else horizon
    alpha_d = s.health_weights[2] * s.trade_off_weights[0]
    tail = [sum(s.discount**j for j in range(k, K)) for k in range(K + 1)]

    incumbent, sequence = _beam(s, K, beam_width)
    best = [incumbent, list(sequence)]
    nodes = [0]
    exhausted = [True]

    def recurse(k: int, state: np.ndarray, total: float, seq: list[int]) -> None:
        nodes[0] += 1
        if max_nodes is not None and nodes[0] > max_nodes:
            exhausted[0] = False
            return
        if k == K:
            if total > best[0]:
                best[0], best[1] = total, list(seq)
            return
        if total - alpha_d * (state[0, 7] / s.N) * tail[k] <= best[0] + 1e-12:
            return
        for action in reversed(range(len(s.betas))):
            nxt = advance(state, s.betas[action], s, s.days)
            seq.append(action)
            recurse(
                k + 1,
                nxt,
                total + (s.discount**k) * float(reward(nxt, action, s)[0]),
                seq,
            )
            seq.pop()

    recurse(0, np.array([s.y0], dtype=float), 0.0, [])
    return best[0], best[1], nodes[0], exhausted[0]


def constant_policies(s: Setting, horizon: int | None = None) -> dict[str, float]:
    """Return the return of holding each action for the whole horizon."""
    K = s.horizon if horizon is None else horizon
    names = {0: "no restriction", 1: "social distancing", 2: "lockdown"}
    return {names.get(a, str(a)): evaluate([a] * K, s)[0] for a in range(len(s.betas))}


def as_digits(actions: Sequence[int]) -> str:
    """Render an action sequence as one digit per decision, as the paper prints it."""
    return "".join(str(a) for a in actions)
