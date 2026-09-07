"""The environments must integrate the probabilities given in the configuration file.

Configuration files spell the key either ``probas`` or ``probs``. Reading only one of the two
made every configuration that used the other spelling fall back to the hard-coded defaults, so
scenarios that differ only in their clinical inputs became the same run. These tests pin the
behaviour down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pandemic_control.environment.seiradh import SEIRADH_Env
from pandemic_control.environment.seiradhv import SEIRADHV_Env

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _probabilities_in_file(path: Path) -> list[float] | None:
    spec = json.loads(path.read_text()).get("spec-params", {})
    values = spec.get("probas", spec.get("probs"))
    return [float(v) for v in values] if values else None


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.json")), ids=lambda p: p.name)
def test_environment_uses_the_configured_probabilities(path: Path) -> None:
    """Whichever spelling a configuration uses, the model integrates those values."""
    expected = _probabilities_in_file(path)
    if expected is None or len(expected) != 4:
        pytest.skip(f"{path.name} declares no four-entry probability list")
    try:
        env = SEIRADHV_Env(str(path))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        # configurations for other models, or incomplete ones
        pytest.skip(f"{path.name} is not a usable SEIRADHV configuration: {error}")
    assert [float(v) for v in env.probas] == expected


def test_both_spellings_are_accepted() -> None:
    """``probas`` and ``probs`` name the same thing."""
    base = json.loads((CONFIGS / "envs_tests" / "default-seiradhv.json").read_text())
    values = [0.83, 0.41, 0.05, 0.11]

    with_a = json.loads(json.dumps(base))
    with_a["spec-params"]["probas"] = values

    without_a = json.loads(json.dumps(base))
    without_a["spec-params"].pop("probas")
    without_a["spec-params"]["probs"] = values

    assert [float(v) for v in SEIRADHV_Env(with_a).probas] == values
    assert [float(v) for v in SEIRADHV_Env(without_a).probas] == values


def test_seiradh_also_reads_the_configuration() -> None:
    """The same fix applies to the SEIRADH stage."""
    config = json.loads((CONFIGS / "envs_tests" / "default-seiradhv.json").read_text())
    config["spec-params"]["probas"] = [0.7, 0.25, 0.03, 0.09]
    assert [float(v) for v in SEIRADH_Env(config).probas] == [0.7, 0.25, 0.03, 0.09]


def test_different_clinical_profiles_give_different_epidemics() -> None:
    """Two profiles that differ only in their probabilities must not produce the same run.

    This is the failure the bug caused: the fragile and resistant configurations were compared
    against each other while both silently used the hard-coded defaults.
    """
    fragile = SEIRADHV_Env(str(CONFIGS / "test_resistance" / "params-fragile.json"))
    resistant = SEIRADHV_Env(str(CONFIGS / "test_resistance" / "params-resistant.json"))
    assert list(fragile.probas) != list(resistant.probas)

    peaks = []
    for env in (fragile, resistant):
        env.reset(seed=0)
        peak_hospital = 0.0
        for _ in range(40):
            observation, *_ = env.step(0)
            peak_hospital = max(peak_hospital, float(observation[5]))
        peaks.append(peak_hospital)

    fragile_peak, resistant_peak = peaks
    assert fragile_peak > 1.5 * resistant_peak
