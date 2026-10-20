"""Offline tests of the reinforcement-learning layer on a linear economy with published coefficients."""

from __future__ import annotations

import math

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
    config = rl.RLConfig(total_timesteps=300, learning_starts=100, eval_episodes=3, q_episodes=10, economies=("SVAR",))
    return rl.PolicyLab(config, {"SVAR": rl.RecursiveLinearTransition(lin.PAPER_TABLE9)}, train)


def test_config_round_trip(tmp_path) -> None:
    config = rl.RLConfig(macro_clamp_pi=float("inf"), actor_node_grid=(2, 3), economies=["SVAR", "NARX"])
    assert rl.RLConfig.load(config.save(tmp_path / "config.json")) == config
    assert config.economies == ("SVAR", "NARX") and config.observes_rate
    assert not rl.RLConfig(omega_di=0.0).observes_rate
    assert config.with_full_grid().critic_node_grid == tuple(range(1, 11))
    with pytest.raises(ValueError):
        rl.RLConfig(gamma=1.5)


def test_reward_matches_formula(lab: rl.PolicyLab) -> None:
    cfg = lab.config
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    assert float(rl.cb_reward(cfg, t(3.0), t(1.0), t(4.0), t(3.0))) == pytest.approx(-(0.5 + 0.5 + 0.5))
    # beyond the 2 pp band the penalty adds 10 x the squared deviation
    assert float(rl.cb_reward(cfg, t(5.0), t(0.0), t(0.0), t(0.0))) == pytest.approx(-(0.5 * 9 + 10 * 9))
    assert float(rl.cb_penalty(cfg, t(5.0), t(0.0))) == pytest.approx(90.0)


def test_environment_and_common_shocks(lab: rl.PolicyLab) -> None:
    lab.check("SVAR")
    env = lab.make_env("SVAR", observation_lag=True)
    assert env.observation_spec["observation"].shape[-1] == 5          # x2 plus i_{t-1}
    td = env.reset()
    assert float(td["observation"][0, 4]) == pytest.approx(float(td["lags"][0, 4]))
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
    assert set(lab.loss(a)) == {"dev2 pi", "dev2 y", "dev2 di", "Loss", "penalty"}
    table = lab.loss_table({"SVAR": lab.simulate_policies("SVAR")})
    assert list(table.index.get_level_values("Policy")) == ["Actual", "TR93", "NPP", "BA"]


def test_reset_lags_follow_the_data(lab: rl.PolicyLab) -> None:
    env = lab.make_env("SVAR", observation_lag=False, with_episodes=False, num_envs=64)
    lags = env.reset()["lags"].numpy()
    data = lab.data.numpy()
    for row in lags:                                     # (y_t, y_{t-1}, pi_t, pi_{t-1}, i_{t-1}, i_{t-2})
        t = int(np.argmin(np.abs(data[:, 0] - row[0])))
        assert t >= 2
        assert np.allclose(row, [data[t, 0], data[t - 1, 0], data[t, 1], data[t - 1, 1], data[t - 1, 2], data[t - 2, 2]])


def test_transmission_check(lab: rl.PolicyLab) -> None:
    response = lab.transmission("SVAR", horizon=8)
    assert list(response.columns) == ["dy", "dpi", "cum_dpi"] and len(response) == 8
    # H&T Table 9: the first-lag rate coefficient in the output-gap equation is positive, inflation falls
    assert response.loc[1, "dy"] == pytest.approx(lin.PAPER_TABLE9.coef_y["i_lag1"], abs=1e-9)
    assert response.loc[8, "dpi"] < 0.0
    table = lab.transmission_table(horizons=(1, 8))
    assert bool(table.loc["SVAR", "Passes"]) and lab.passing_economies() == ("SVAR",)
    lab.require_transmission()


def test_steady_state_reward_is_evaluated_at_the_fixed_point(lab: rl.PolicyLab) -> None:
    rule = lab.reference_rules()["TR93"]
    ss = lab.steady_state(rule, "SVAR", observation_lag=False, observe_rate=False)
    assert ss is not None
    pi, y, i = ss
    assert i == pytest.approx(max(0.0, 1.0 + 1.5 * pi + 0.5 * y), abs=1e-6)
    reward = lab.steady_state_reward(rule, "SVAR", observation_lag=False, observe_rate=False)
    expected = float(rl.cb_reward(lab.config, torch.tensor(pi), torch.tensor(y), torch.tensor(i), torch.tensor(i)))
    assert reward == pytest.approx(expected)
    assert math.isfinite(lab.evaluation_reward(rule, "SVAR", observation_lag=False, observe_rate=False, episodes=2))

    class Explosive(torch.nn.Module):                     # i = 15 whenever pi > 2, 0 otherwise: no fixed point
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.where(obs[..., 1:2] > 2.0, 1.0, -1.0).to(torch.float32)

    assert lab.steady_state_reward(Explosive(), "SVAR", observation_lag=False, observe_rate=False) == -math.inf


def test_actor_zero_lower_bound_inside_the_actor() -> None:
    actor = rl.LinearActorNet(2)
    with torch.no_grad():
        actor.mu.weight.fill_(-5.0)
        actor.mu.bias.zero_()
    out = actor(torch.tensor([[3.0, 3.0], [0.0, 0.0]]))
    assert float(out[0, 0]) == pytest.approx(-1.0 + rl.ZLB_LEAK * (-30.0 + 1.0)) and float(out[1, 0]) == 0.0
    # below the bound the environment's clip gives a rate of exactly zero, but a gradient survives
    assert float(rl.normalised_to_rate(out[0], 15.0)) == 0.0
    x = torch.tensor([[-2.0]], requires_grad=True)
    rl.zero_lower_bound(x).sum().backward()
    assert float(x.grad) == pytest.approx(rl.ZLB_LEAK)


def test_episode_tracker_filter(lab: rl.PolicyLab) -> None:
    from tensordict import TensorDict

    actor = rl.LinearActorNet(2)
    tracker = rl.EpisodeTracker(actor, lab.config)

    def frames(ER: float, ES: int, terminated: bool) -> TensorDict:
        return TensorDict({"next": TensorDict({
            "done": torch.tensor([[True]]), "terminated": torch.tensor([[terminated]]),
            "episode_reward": torch.tensor([[ER]]), "step_count": torch.tensor([[ES]])}, batch_size=[1])}, batch_size=[1])

    tracker.observe(frames(-10.0, 5, True), checkpoint=False)        # warm-up: counted, never checkpointed
    tracker.observe(frames(-10.0, 5, True))                          # qualifies: ER/ES = -2 > -4, in band before T
    tracker.observe(frames(-10.0, lab.config.t_max, False))          # truncated: excluded
    tracker.observe(frames(-30.0, 5, True))                          # ER/ES = -6: excluded
    tracker.observe(frames(-1.0, 1, True))                           # one step: excluded
    assert len(tracker.episode_rewards) == 5 and len(tracker.candidates) == 1
    assert tracker.candidates[0].episode == 2


def test_tabular_policy_bins_and_fallback() -> None:
    policy = rl.TabularQPolicy(15.0)
    assert policy.bins == (len(rl.Y_EDGES) + 1, len(rl.PI_EDGES) + 1) and policy.action_grid.numel() == 31
    iy, ipi = policy.discretise(torch.tensor([[0.0, 2.0], [-10.0, 10.0]]))
    assert iy.tolist() == [4, 0] and ipi.tolist() == [3, 7]
    # unvisited state: the fallback rule's nearest grid rate (TR93 at pi = 2, y = 0 is 4 percent)
    rate = rl.normalised_to_rate(policy(torch.tensor([[0.0, 2.0]])).double(), 15.0)
    assert float(rate) == pytest.approx(4.0)
    # a visited state picks the best visited action even if an unvisited one still has Q = 0
    policy.Q[4, 3, 10], policy.N[4, 3, 10] = -1.0, 1
    policy.Q[4, 3, 12], policy.N[4, 3, 12] = -3.0, 1
    assert int(policy.greedy_index(torch.tensor([[0.0, 2.0]]))) == 10


def test_training_and_cell_round_trip(lab: rl.PolicyLab, tmp_path) -> None:
    ddpg, log = rl.train_ddpg_cell(lab, "SVAR", "x1", "linear")
    assert len(log) == len(lab.config.critic_node_grid) and ddpg.label == "DDPG x1 linear"
    assert set(log.columns) == {"Critic nodes", "Actor nodes", "SS reward", "Eval reward", "Candidates", "Episodes"}
    cells = [ddpg, rl.train_sac_cell(lab, "SVAR", "x2"), rl.train_q_cell(lab, "SVAR")]
    assert [c.obs_dim for c in cells] == [3, 5, 2] and cells[2].policy.N.sum() > 0
    rl.save_cells(cells, tmp_path)
    loaded = {cell.algo: cell for cell in rl.load_cells(tmp_path)}
    for cell in cells:
        obs = torch.randn(9, cell.obs_dim)
        rate = cell.rate(obs)
        assert torch.allclose(rate, loaded[cell.algo].rate(obs)) and float(rate.min()) >= 0.0
        assert float(rate.max()) <= lab.config.action_high
        assert loaded[cell.algo].lagged_rate == cell.lagged_rate
    fit = rl.linearise_policy(ddpg)
    assert set(fit) == {"alpha0", "beta_pi", "beta_pi_1", "beta_y", "beta_y_1", "R2"} and fit["beta_pi_1"] == 0.0
    assert rl.linearise_policy(cells[1])["R2"] <= 1.0
    table = rl.coefficient_table(cells)
    assert table.index.get_level_values("Economy")[0] == "Reference rules" and r"$\beta_\pi^1$" in table.columns
    assert lab.loss_table({"SVAR": lab.simulate_policies("SVAR", cells)}).shape[0] == 7


def test_rs99_published_model() -> None:
    table = rl.rs99_table({"TR93": {"beta_pi_0": 1.5, "beta_y_0": 0.5}, "passive": {"beta_pi_0": 0.2}})
    assert table.loc["TR93", ["var(pi)", "var(y)", "var(Delta i)", "Mean"]].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert np.isinf(table.loc["passive", "Mean"]) and table.loc["passive", "rho"] >= 1.0
    F = rl.build_rs99_F({"beta_pi_0": 1.5, "beta_y_0": 0.5})
    assert F.shape == (9, 9) and torch.allclose(F[2, 2:6], torch.tensor([0.70, -0.10, 0.28, 0.12], dtype=torch.float64))
    # the four-quarter average rate: one quarter of i_t = K z_t plus one quarter of each stored lag
    assert F[0, 6] == pytest.approx(-0.10 / 4) and F[0, 0] == pytest.approx(1.16 - 0.10 * 0.5 / 4)
    assert F[0, 2] == pytest.approx(-0.10 * 1.5 / 4 + 0.10 / 4)
    Q = torch.eye(9, dtype=torch.float64)
    V = rl.solve_discrete_lyapunov(F, Q)
    assert torch.allclose(V, F @ V @ F.T + Q, atol=1e-9)
    # the Taylor principle: with a unit-root Phillips curve a rule with beta_pi < 1 is unstable
    assert rl.rs99_variances({"beta_pi_0": 0.9, "beta_y_0": 0.5}) is None
    assert rl.rs99_variances({"beta_pi_0": 1.1, "beta_y_0": 0.5}) is not None
