"""Offline tests of the non-linear economies on a simulated panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from autofed import ArtifactError  # noqa: E402
from autofed import nonlinear as nl  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    index = pd.period_range("1987Q3", "2012Q2", freq="Q", name="quarter")
    y, pi, i = np.zeros(len(index)), np.full(len(index), 2.0), np.full(len(index), 4.0)
    for t in range(2, len(index)):
        y[t] = 0.9 * y[t - 1] - 0.05 * (i[t - 1] - 4.0) + 0.4 * rng.standard_normal()
        pi[t] = 2.0 + 0.85 * (pi[t - 1] - 2.0) + 0.1 * y[t - 1] + 0.2 * rng.standard_normal()
        i[t] = max(0.8 * i[t - 1] + 0.2 * (1.0 + 1.5 * pi[t] + 0.5 * y[t]), 0.05)
    return pd.DataFrame({"y": y, "pi": pi, "i": i}, index=index)


def test_scaler_round_trip() -> None:
    x = torch.tensor([[1.0, -2.0], [3.0, 5.0], [2.0, 0.0]], dtype=torch.float64)
    scaler = nl.MapMinMaxScaler().fit(x)
    scaled = scaler.transform(x)
    assert float(scaled.min()) == -1.0 and float(scaled.max()) == 1.0
    assert torch.allclose(scaler.inverse_transform(scaled), x)


def test_narx_search_predict_and_round_trip(frame: pd.DataFrame, tmp_path) -> None:
    train = frame.iloc[:80]
    design = nl.build_restricted_narx_design_matrices(train)
    assert design.X_y.shape == (78, 3) and design.X_pi.shape == (78, 6)
    kwargs = dict(time_index=design.time_index, hidden_min=1, hidden_max=2, n_trials=2, max_epochs=40)
    search_y = nl.run_hidden_unit_search(equation_name="Output gap", X=design.X_y, y=design.y_target, base_seed=1, **kwargs)
    search_pi = nl.run_hidden_unit_search(equation_name="Inflation", X=design.X_pi, y=design.pi_target, base_seed=2, **kwargs)
    economy = nl.NARXEconomy.from_search(search_y, search_pi)
    fit = economy.predict(train)
    economy.set_sigma(fit)
    assert fit.index[0] == pd.Period("1988Q1", "Q") and np.isfinite(fit.to_numpy()).all()
    assert float(((fit["y"] - fit["y_hat"]) ** 2).mean()) < float(train["y"].var())      # beats the mean
    loaded = nl.NARXEconomy.load(economy.save(tmp_path / "narx.pt"))
    pd.testing.assert_frame_equal(loaded.predict(frame), economy.predict(frame))
    assert (loaded.sigma_y, loaded.sigma_pi) == (economy.sigma_y, economy.sigma_pi)
    with pytest.raises(ArtifactError):
        nl.NSSMEconomy.load(tmp_path / "narx.pt")


def test_nssm_training_anchor_and_round_trip(frame: pd.DataFrame, tmp_path) -> None:
    train = frame.iloc[:80]
    result = nl.train_nssm(nl.build_full_nssm_inputs(train), epochs=60, seed=3)
    assert result.loss_history["total"].iloc[-1] < result.loss_history["total"].iloc[0]
    economy = nl.NSSMEconomy.from_training(result, train)
    fit, states_y, _ = economy.run(frame)
    assert states_y.shape == (len(frame) - 2, 4) and economy.sigma_y > 0
    # the anchored state is the state reached at the end of the estimation window
    assert torch.equal(economy.terminal_y, states_y[77:78])
    loaded = nl.NSSMEconomy.load(economy.save(tmp_path / "nssm.pt"))
    pd.testing.assert_frame_equal(loaded.predict(frame), fit)
    assert torch.equal(loaded.terminal_pi, economy.terminal_pi)
    with pytest.raises(ArtifactError):
        nl.NARXEconomy.load(tmp_path / "missing.pt")
