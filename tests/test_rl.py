"""Offline tests of the reinforcement-learning layer on a linear economy with published coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from autofed import linear as lin  # noqa: E402
from autofed import rl  # noqa: E402


@pytest.fixture(scope="module")
def lab() -> rl.PolicyLab:
    rng = np.random.default_rng(3)
    index = pd.period_range("1987Q3", "2007Q2", freq="Q", name="quarter")
    train = pd.DataFrame({"y": rng.normal(0.0, 1.0, len(index)), "pi": rng.normal(2.5, 0.8, len(index)),
                          "i": rng.uniform(1.0, 8.0, len(index))}, index=index)
    config = rl.RLConfig(total_timesteps=300, learning_starts=100, ss_rollout_steps=20, q_episodes=10)
    return rl.PolicyLab(config, {"SVAR": rl.RecursiveLinearTransition(lin.PAPER_TABLE9)}, train)


def test_config_round_trip(tmp_path) -> None:
    config = rl.RLConfig(macro_clamp_pi=float("inf"), actor_node_grid=(2, 3))
    assert rl.RLConfig.load(config.save(tmp_path / "config.json")) == config
    assert config.with_full_grid().critic_node_grid == tuple(range(1, 11))
    with pytest.raises(ValueError):
        rl.RLConfig(gamma=1.5)


def test_reward_matches_formula(lab: rl.PolicyLab) -> None:
    cfg = lab.config
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    assert float(rl.cb_reward(cfg, t(3.0), t(1.0), t(4.0), t(3.0))) == pytest.approx(-(0.5 + 0.5 + 0.5))
    # beyond the 2 pp band the penalty adds 10 x the squared deviation
    assert float(rl.cb_reward(cfg, t(5.0), t(0.0), t(0.0), t(0.0))) == pytest.approx(-(0.5 * 9 + 10 * 9))


def test_environment_and_common_shocks(lab: rl.PolicyLab) -> None:
    lab.check("SVAR")
    rules = lab.reference_rules()
    a = lab.counterfactual("SVAR", rules["TR93"], observation_lag=False)
    b = lab.counterfactual("SVAR", rules["BA"], observation_lag=True)
    assert a.index.equals(lab.train.index) and np.allclose(a[["y", "pi"]].iloc[:2], lab.train[["y", "pi"]].iloc[:2])
    assert a["i"].iloc[0] == lab.train["i"].iloc[0]
    # data timing: the rate of each quarter is the rule applied to that quarter's inflation and output gap
    assert np.allclose(a["i"].iloc[1:], np.clip(1.0 + 1.5 * a["pi"].iloc[1:] + 0.5 * a["y"].iloc[1:], 0.0, 15.0), atol=1e-4)
    # both policies face the same shocks: the output-gap residuals implied by the two paths coincide
    shock = lambda p: lin.structural_residuals(p, lin.PAPER_TABLE9)["y"].to_numpy()[2:]  # noqa: E731
    assert np.allclose(shock(a), shock(b), atol=1e-9) and not np.allclose(a["i"].iloc[2:], b["i"].iloc[2:])
    assert set(lab.loss(a)) == {"dev2 pi", "dev2 y", "dev2 di", "Loss"}
    table = lab.loss_table({"SVAR": lab.simulate_policies("SVAR")})
    assert list(table.index.get_level_values("Policy")) == ["Actual", "TR93", "NPP", "BA"]


def test_training_and_cell_round_trip(lab: rl.PolicyLab, tmp_path) -> None:
    ddpg, log = rl.train_ddpg_cell(lab, "SVAR", "x1", "linear")
    assert len(log) == len(lab.config.critic_node_grid) and ddpg.label == "DDPG x1 linear"
    cells = [ddpg, rl.train_sac_cell(lab, "SVAR", "x2"), rl.train_q_cell(lab, "SVAR")]
    rl.save_cells(cells, tmp_path)
    loaded = {cell.algo: cell for cell in rl.load_cells(tmp_path)}
    for cell in cells:
        obs = torch.randn(9, rl.OBS_DIM[cell.obs_spec])
        rate = cell.rate(obs)
        assert torch.allclose(rate, loaded[cell.algo].rate(obs)) and float(rate.min()) >= 0.0
        assert float(rate.max()) <= lab.config.action_high
    assert set(rl.linearise_policy(ddpg)) == {"alpha0", "beta_pi", "beta_y", "R2"}
    assert rl.coefficient_table(cells).index.get_level_values("Economy")[0] == "Reference rules"


def test_rs99_ratios() -> None:
    table = rl.rs99_table({"TR93": {"beta_pi_0": 1.5, "beta_y_0": 0.5}, "passive": {"beta_pi_0": 0.2}})
    assert table.loc["TR93"].tolist() == [1.0, 1.0, 1.0, 1.0] and np.isinf(table.loc["passive", "Mean"])
    F = rl.build_rs99_F({"beta_pi_0": 1.5, "beta_y_0": 0.5})
    Q = torch.eye(6, dtype=torch.float64)
    V = rl.solve_discrete_lyapunov(F, Q)
    assert torch.allclose(V, F @ V @ F.T + Q, atol=1e-9)
