# Conformal error bars for learning-based quantum error mitigation

Code and data for the paper

> An Cong Tran and Van Lam Le. *Conformal Error Bars for Learning-Based Quantum Error Mitigation under Circuit Shift and Noise Drift.* Submitted to *Quantum Machine Intelligence* (2026).

The repository contains:

- the simulation pipeline that generates the datasets;
- the datasets used in the paper;
- the scripts that reproduce every table and figure.

All experiments run on a CPU. The paper's results were produced on a 2018 MacBook Pro (4-core Intel i7, 16 GB RAM).

## Contents

```
Quantum-Error-Mitigation/
├── README.md
├── LICENSE
├── requirements.txt
├── generate_data.sh          # regenerate all datasets from scratch (optional, hours)
├── reproduce_analysis.sh     # reproduce all tables and figures from data/ (no simulation)
├── src/cpqem/                # simulation library
│   ├── circuits.py           #   TFIM and XXZ Trotter circuits, basis compilation
│   ├── clifford.py           #   near-Clifford circuits, non-Clifford gate count k
│   ├── noise.py              #   noise models from IBM fake backends (+ coherent errors)
│   ├── simulate.py           #   folding, exact density-matrix expectations, shot sampling
│   └── features.py           #   circuit and observable features
├── scripts/
│   ├── make_dataset.py       # data: target + near-Clifford circuits (k = 0, 2, 4, 8)
│   ├── make_kladder.py       # data: k-ladder circuits (k = 16, 32) for existing targets
│   ├── make_drift.py         # data: drift streams, 5 scenarios
│   ├── verify_blocks.py      # sanity checks of the simulation pipeline
│   ├── g0_split_cp.py        # Sec. 6.1  split CP baseline
│   ├── p3_weighted_cp.py     # Sec. 6.2  conformal methods under circuit shift, k-scaled CP
│   ├── fig_rq2.py            # Figs. 2, 3, B1
│   ├── p6_baselines.py       # Sec. 6.3  existing error bars, alpha sensitivity, circuit-level CP
│   ├── a2_diagnostic.py      # Sec. 6.4  A2 check on the k-ladder (Fig. 4b)
│   ├── fig_heldout.py        # Fig. 4
│   ├── p7_ladder.py          # Appendix C  pre-registered ladder variant
│   ├── p4_drift.py           # Sec. 6.5  noise drift, Table 6, Fig. 5
│   └── p5_shots.py           # Sec. 6.6  shot budget, Fig. 6
├── tests/test_pipeline.py    # unit tests of the simulation pipeline
├── data/                     # datasets used in the paper (41 MB, gzip CSV)
├── manuscript_figures/       # TikZ source of Fig. 1
└── results/                  # created by the scripts (not tracked)
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/tcanvngmail/Quantum-Error-Mitigation.git
cd Quantum-Error-Mitigation
python -m venv .venv && source .venv/bin/activate      # or a conda environment
pip install -r requirements.txt
pytest -q tests/                                        # 9 tests, a few seconds
```

The paper used Qiskit 1.x and Qiskit Aer 0.15 or later. For better reproducibility, record your environment with `pip freeze > environment.txt`.

## Quick start: reproduce the paper from the released data

```bash
./reproduce_analysis.sh
```

The script needs no quantum simulation. It runs all analyses on the files in `data/` and writes:

- tables as CSV files to `results/`;
- figures as PDF and PNG files to `results/`;
- the printed tables to `results/logs/`.

On a 4-core laptop it takes about 1–2 hours, mostly for the 50-split runs of Sec. 6.2. Every step can also be run on its own (see below).

All random splits use fixed seeds (`2000 + repetition`), so the numbers should match the paper. Small differences in the third decimal can come from different versions of scikit-learn.

## Step-by-step reproduction

The commands below assume:

```bash
export OMP_NUM_THREADS=1
MAIN="data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz data/main_n10.csv.gz data/kladder_n10.csv.gz"
```

### Sec. 6.1: split CP on near-Clifford calibration data

```bash
python scripts/g0_split_cp.py --data data/main_n4-8.csv.gz --reps 50
```

### Sec. 6.2: circuit shift and k-scaled CP (Table 2, Figs. 2, 3, B1)

```bash
python scripts/p3_weighted_cp.py --data $MAIN --reps 50 --model linear --dump
python scripts/p3_weighted_cp.py --data $MAIN --reps 50 --model gbm --dump
python scripts/fig_rq2.py --data $MAIN
```

`--model linear` is vnCDR and `--model gbm` is gradient boosting. The method `kscale_auto` in the output is k-scaled CP as described in the paper. The figures are written to `results/figs/`.

### Sec. 6.3: comparison with existing error bars, sensitivity to α (Tables 3, 4)

```bash
python scripts/p6_baselines.py --data $MAIN --reps 30 --alphas 0.05 0.1 0.2 --out results/p6_main
python scripts/p3_weighted_cp.py --data $MAIN --reps 30 --model gbm --alpha 0.05 --out results/p3_gbm_a0.05
python scripts/p3_weighted_cp.py --data $MAIN --reps 30 --model gbm --alpha 0.2  --out results/p3_gbm_a0.2
```

| Name in the output | Method in the paper |
|---|---|
| `shot_prop` | shot propagation |
| `zne` | ZNE ± shot error |
| `cdr_3sigma` | CDR ±3σ |
| `cdr_gauss` | Gaussian ±zσ |
| `bayes` | Bayesian linear regression |
| `bootstrap` | bootstrap |
| `cv_plus` | CV+ |
| `split_cp` | split CP with k ≤ 8 |
| `kscale` | k-scaled CP |
| `kscale_circ` | circuit-level k-scaled CP |

### Sec. 6.4: held-out XXZ family and coherent noise (Table 5, Fig. 4)

Run the following for each held-out dataset. The example uses XXZ; replace `xxz` with `coh` for coherent error A or `coh2` for coherent error B.

```bash
D="data/xxz_n4-8.csv.gz data/xxz_kladder_n4-8.csv.gz"
python scripts/p3_weighted_cp.py --data $D --reps 30 --model linear --out results/p3_xxz_linear
python scripts/p3_weighted_cp.py --data $D --reps 30 --model gbm    --out results/p3_xxz_gbm
python scripts/p6_baselines.py   --data $D --reps 30 --alphas 0.1   --out results/p6_xxz
python scripts/a2_diagnostic.py  --data $D                       # A2 check on the ladder
python scripts/fig_heldout.py --out results/fig_heldout.pdf
```

`fig_heldout.py` plots the summary numbers printed by the commands above; they are written in the script.

### Appendix C: pre-registered ladder variant

```bash
python scripts/p7_ladder.py --data data/coh2_n4-8.csv.gz data/coh2_kladder_n4-8.csv.gz --model linear
python scripts/p7_ladder.py --data data/coh2_n4-8.csv.gz data/coh2_kladder_n4-8.csv.gz --model gbm
```

The variant evaluated in the paper is `hi_extrap`. Its design was fixed on the coherent A data before the coherent B data were generated.

### Sec. 6.5: noise drift (Table 6, Fig. 5)

```bash
python scripts/p4_drift.py --base data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz \
    --drift data/drift_n4-8.csv.gz --seeds 10
```

The policies `static`, `periodic`, `window`, `aci`, `aci_mart`, `refit` and `track` are those of Table 6. The script writes `p4_summary.csv`, `p4_alarms.csv` (martingale alarms), `p4_rolling.csv` and `p4_rolling.pdf`.

### Sec. 6.6: shot budget (Fig. 6)

```bash
python scripts/p5_shots.py --data $MAIN --reps 30
```

Shots are resampled offline from the exact noisy expectation values, so no simulation is needed.

## Regenerating the datasets (optional)

```bash
./generate_data.sh
```

This runs the density-matrix simulations for all datasets (Table 1 of the paper). It takes several hours per dataset on a 4-core laptop; n = 10 is the slowest. On macOS, prefix the long runs with `caffeinate -i` to keep the machine awake.

Random circuit parameters depend on `--seed`, so regenerated files reproduce the paper's results statistically but not row by row. Use the files in `data/` for exact reproduction.

Noise parameters come from the Qiskit fake backends `FakeSherbrooke` (main) and `FakeBrisbane` (device swap in scenario S5).

## Data description

| File | Content | Rows | Circuits |
|---|---|---|---|
| `main_n4-8.csv.gz` | TFIM, n = 4, 6, 8; targets + near-Clifford (k = 0, 2, 4, 8) | 50,490 | 4,590 |
| `kladder_n4-8.csv.gz` | k-ladder (k = 16, 32) for the targets above | 47,520 | 4,320 |
| `main_n10.csv.gz` | TFIM, n = 10 | 29,070 | 1,530 |
| `kladder_n10.csv.gz` | k-ladder for n = 10 | 27,360 | 1,440 |
| `xxz_n4-8.csv.gz`, `xxz_kladder_n4-8.csv.gz` | held-out XXZ family | 50,490 / 47,520 | 4,590 / 4,320 |
| `coh_n4-8.csv.gz`, `coh_kladder_n4-8.csv.gz` | TFIM + coherent error A (0.03, 0.05 rad) | 50,490 / 47,520 | 4,590 / 4,320 |
| `coh2_n4-8.csv.gz`, `coh2_kladder_n4-8.csv.gz` | TFIM + coherent error B (0.05, 0.08 rad) | 50,490 / 47,520 | 4,590 / 4,320 |
| `drift_n4-8.csv.gz` | drift streams, scenarios S0, S2–S5, 300 time steps | 148,500 | 13,500 |

Each row is one (circuit, observable) pair. The main columns are listed below.

| Column | Meaning |
|---|---|
| `group_id` | target circuit and all near-Clifford circuits derived from it |
| `circuit_id`, `kind` | circuit identifier; `target` or `nearclifford` |
| `n`, `steps`, `J`, `h`, `dt` | qubits, Trotter steps, Hamiltonian parameters (for XXZ, `h` holds Δ), time step |
| `n_keep`, `n_nonclifford` | number of non-Clifford gates k |
| `nc_mode` | near-Clifford sampling rule (`sample` or `mixed`) |
| `depth`, `n_cx`, `n_rz`, `n_1q_phys` | circuit structure |
| `dev_*` | mean device parameters (T1, T2, gate and readout errors, coherent errors) |
| `obs`, `obs_weight`, `obs_pos` | observable (Z_i or Z_i Z_{i+1}), Pauli weight, position |
| `y_ideal` | ideal (noiseless) expectation value: the label |
| `exact_s1`, `exact_s3`, `exact_s5` | exact noisy expectation value at noise scale 1, 3, 5 |
| `noisy_s1`, `noisy_s3`, `noisy_s5` | the same with 2000-shot sampling noise (model inputs) |
| `se_s1`, `se_s3`, `se_s5` | shot standard errors |
| `scenario`, `t`, `role`, `t_scale`, `err_scale`, `backend` | drift files only: scenario, time step, probe/target, noise scale factors, backend |

## Citation

If you use this code or data, please cite the paper (details will be added after publication).

## License

The code is released under the MIT License (see `LICENSE`). The data are released under CC BY 4.0.
