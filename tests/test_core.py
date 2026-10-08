"""Offline tests on a simulated panel (no FRED access)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import autofed as af
from autofed import linear as lin


@pytest.fixture(scope="module")
def panel() -> af.MacroPanel:
    rng = np.random.default_rng(11)
    index = pd.period_range("1976Q3", "2026Q2", freq="Q", name="quarter")
    n = len(index)
    y, pi, i = np.zeros(n), np.zeros(n), np.zeros(n)
    y[:2], pi[:2], i[:2] = (-0.5, -0.3), (5.5, 5.3), (7.0, 6.9)
    for t in range(2, n):
        y[t] = 0.31 + 0.93 * y[t - 1] - 0.02 * pi[t - 1] - 0.05 * i[t - 1] - 0.03 * i[t - 2] + 0.48 * rng.standard_normal()
        pi[t] = (0.13 - 0.064 * y[t] + 0.209 * y[t - 1] - 0.109 * y[t - 2] + 1.25 * pi[t - 1] - 0.30 * pi[t - 2]
                 - 0.017 * i[t - 1] + 0.185 * rng.standard_normal())
        i[t] = max(0.85 * i[t - 1] + 0.15 * (1.0 + 1.5 * pi[t] + 0.5 * y[t]) + 0.25 * rng.standard_normal(), 0.07)
    potential = 6000 * np.exp(0.007 * np.arange(n))
    frame = pd.DataFrame({"y": y, "pi": pi, "i": i, "gdp_real": potential * (1 + y / 100), "gdp_potential": potential},
                         index=index)
    return af.MacroPanel(frame, retrieved="2026-01-01")


def test_panel_windows_and_round_trip(panel: af.MacroPanel, tmp_path) -> None:
    assert len(panel.train) == 80 and panel.train.index[0] == pd.Period("1987Q3", "Q")
    assert panel.tvp.index[42] == pd.Period("1987Q3", "Q")
    loaded = af.load_macro_panel(panel.save(tmp_path / "data"))
    pd.testing.assert_frame_equal(loaded.data, panel.data, check_exact=False, rtol=1e-9)
    with pytest.raises(af.ArtifactError):
        af.load_macro_panel(tmp_path / "missing")


def test_panel_rejects_gaps(panel: af.MacroPanel) -> None:
    with pytest.raises(af.DataError):
        af.MacroPanel(panel.data.drop(panel.data.index[100]))


def test_rules_level_form() -> None:
    assert af.rules.TR93.alpha0 == 1.0 and af.rules.NPP.alpha0 == 0.0 and af.rules.BA.alpha0 == 1.0
    assert af.rules.TR93.prescribe(3.0, 0.5) == pytest.approx(1 + 1.5 * 3.0 + 0.5 * 0.5)
    assert af.rules.NPP.prescribe(-3.0, 0.0, zlb=True) == 0.0


def test_svar_spec_and_counterfactual_identity(panel: af.MacroPanel, tmp_path) -> None:
    fit = lin.fit_linear_svar(panel.train)
    spec = fit.to_spec()
    assert lin.RecursiveLinearSpec.load(spec.save(tmp_path / "spec.json")).to_dict() == spec.to_dict()
    pred = spec.predict(panel.train)
    assert np.allclose(pred["y_hat"], fit.mod_y.fittedvalues) and np.allclose(pred["pi_hat"], fit.mod_pi.fittedvalues)
    # the realised policy path with the historical shocks must reproduce the data
    assert np.allclose(lin.simulate(panel.full, spec, None), panel.full)
    assert np.isfinite(lin.simulate(panel.full, lin.PAPER_TABLE9, af.rules.TR93).to_numpy()).all()
    with pytest.raises(af.SpecificationError):
        lin.RecursiveLinearSpec("bad", {"nope": 1.0}, {}, 1.0, 1.0)


def test_closed_loop_radius_of_published_estimates() -> None:
    radii = {name: lin.closed_loop_radius(lin.PAPER_TABLE9, rule) for name, rule in af.REFERENCE_RULES.items()}
    assert radii == pytest.approx({"TR93": 0.9084, "NPP": 0.9035, "BA": 0.9335}, abs=5e-4)


def test_session_numbers_and_files(panel: af.MacroPanel, tmp_path) -> None:
    nb = af.NotebookSession("07_demo", number=7, workspace=af.Workspace(tmp_path))
    fig, ax = plt.subplots()
    ax.plot(panel.train["y"].to_numpy())
    first = nb.figure(fig, "output_gap", "Output gap", show=False)
    table = nb.table(lin.table_nine(lin.fit_linear_svar(panel.train)), "estimates", "Estimates", show=False)
    again = nb.figure(plt.figure(), "output_gap", "Output gap", show=False)          # re-run keeps the number
    assert (first.label, table.label, again.label) == ("Figure 7.1", "Table 7.1", "Figure 7.1")
    assert first.file == "fig_07_01_output_gap.png" and (nb.figure_dir / first.file).is_file()
    assert (nb.figure_dir / "tab_07_01_estimates.csv").is_file() and (nb.figure_dir / "manifest.json").is_file()
    with pytest.raises(ValueError):
        nb.figure(plt.figure(), "Bad Slug", "x", show=False)
