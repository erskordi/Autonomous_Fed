# Notebooks

Research notebooks of the Autonomous Fed project. Reusable code lives in the `autofed` package
(`src/autofed`); a notebook holds the narrative, the configuration and the exhibits.

## Run order

Each notebook loads what earlier ones saved under `artifacts/`, so run them in order the first time.

| # | Notebook | Reads | Writes to `artifacts/` |
|---|---|---|---|
| 0 | `00_data.ipynb` | FRED (key in `FRED_API_KEY`) | `data/macro_panel.{csv,json}` |
| 1 | `01_linear_svar.ipynb` | `data/` | `linear/svar_restricted.json`, `linear/one_step_fixed.csv` |
| 2 | `02_linear_rolling_expanding.ipynb` | `data/`, `linear/one_step_fixed.csv` | `linear/one_step_rolling.csv`, `linear/one_step_expanding.csv` |
| 3 | `03_linear_tvp_svar_sv.ipynb` | `data/`, the one-step fits of 1 and 2 | `linear/tvp_anchor.json`, `linear/one_step_tvp.csv`, `linear/oos_summary.csv` |
| 4 | `04_policy_rules.ipynb` | `data/`, `linear/svar_restricted.json` | `rules/static_prescriptions.csv`, `rules/counterfactual_<rule>.csv` |
| 5 | `05_nonlinear_narx.ipynb` | `data/`, `linear/one_step_fixed.csv` | `nonlinear/narx.pt`, `nonlinear/one_step_narx.csv` |
| 6 | `06_nonlinear_nssm.ipynb` | `data/`, the one-step fits of 1 and 5 | `nonlinear/nssm.pt`, `nonlinear/one_step_nssm.csv`, `nonlinear/oos_summary.csv` |
| 7 | `07_rl_environments.ipynb` | `data/`, the four saved economies | `rl/config.json` |
| 8 | `08_rl_ddpg.ipynb` | `rl/config.json`, the four economies | `rl/ddpg/*.pt` |
| 9 | `09_rl_sac.ipynb` | `rl/config.json`, the four economies | `rl/sac/*.pt` |
| 10 | `10_rl_qlearning.ipynb` | `rl/config.json`, the four economies | `rl/qlearning/*.pt` |
| 11 | `11_rl_robustness_comparison.ipynb` | the trained policies of 8 to 10 | nothing (comparison only) |

Notebooks 8 to 10 are independent of each other once 7 has run. Every reinforcement-learning setting is
defined in notebook 7 and loaded by the others.

## Setup

```bash
pip install -e ".[notebooks,rl]"   # from the repository root; 05-06 need PyTorch, 07-11 also TorchRL
export FRED_API_KEY=...              # only notebook 00 needs it
```

## Conventions

- **Exhibits.** Every figure and table goes through `nb.figure(...)`, `nb.table(...)` or
  `nb.listing(...)`. They are numbered per notebook in order of appearance (`Figure 3.2` is the second
  figure of notebook 3) and saved to `figures/<notebook>/` as `fig_NN_kk_<slug>.png` or
  `tab_NN_kk_<slug>.png`, with the table's numbers in a `.csv` (or `.txt`) of the same name and a
  `manifest.json` listing all exhibits. Re-running a cell keeps its number.
- **Documents.** The last cell of each notebook writes `documents/<notebook>.pdf`. It converts the copy
  saved on disk: with auto-save on, it waits for the save; otherwise it writes nothing and asks you to
  save and run the cell again. `webpdf` needs `playwright install chromium`; `latex` needs pandoc and
  XeLaTeX.
- **Data vintage.** Only notebook 00 downloads data. Re-run it to refresh the vintage, then re-run the
  rest.

## Layout

```
notebooks/
  00_data.ipynb ... 11_rl_robustness_comparison.ipynb
  figures/<notebook>/     figures, table images, manifest
  documents/              one PDF per notebook
  artifacts/              data and fitted models passed between notebooks (regenerable)
  archive/                superseded notebooks and their PDFs
```
