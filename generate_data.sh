#!/bin/bash
# Regenerate all datasets from scratch (optional: the released files in data/ are the ones used in the paper).
# Runtime on a 4-core laptop: several hours per dataset; n = 10 is the slowest.
# Random draws depend on --seed, so regenerated files match the paper statistically, not row by row.
set -e
export OMP_NUM_THREADS=1
J=4   # number of parallel worker processes

# Main TFIM data (n = 4-8 and n = 10), with the k-ladder
python scripts/make_dataset.py --family tfim --n-list 4 6 8 --steps-list 4 8 12 --targets-per-config 30 \
    --nc-per-target 16 --shots 2000 --jobs $J --out data/main_n4-8.csv.gz
python scripts/make_dataset.py --family tfim --n-list 10 --steps-list 4 8 12 --targets-per-config 30 \
    --nc-per-target 16 --shots 2000 --jobs $J --out data/main_n10.csv.gz
python scripts/make_kladder.py --base data/main_n4-8.csv.gz --k-list 16 32 --per-k 8 --jobs $J --out data/kladder_n4-8.csv.gz
python scripts/make_kladder.py --base data/main_n10.csv.gz --k-list 16 32 --per-k 8 --jobs $J --out data/kladder_n10.csv.gz

# Drift streams (5 scenarios, 300 time steps)
python scripts/make_drift.py --T 300 --n-list 4 6 8 --jobs $J --out data/drift_n4-8.csv.gz

# Held-out XXZ family
python scripts/make_dataset.py --family xxz --n-list 4 6 8 --steps-list 2 4 6 --targets-per-config 30 \
    --nc-per-target 16 --shots 2000 --jobs $J --seed 55 --out data/xxz_n4-8.csv.gz
python scripts/make_kladder.py --base data/xxz_n4-8.csv.gz --k-list 16 32 --per-k 8 --jobs $J --out data/xxz_kladder_n4-8.csv.gz

# Coherent noise A (0.03, 0.05 rad) and B (0.05, 0.08 rad)
python scripts/make_dataset.py --family tfim --coh-rx 0.03 --coh-zz 0.05 --n-list 4 6 8 --steps-list 4 8 12 \
    --targets-per-config 30 --nc-per-target 16 --shots 2000 --jobs $J --seed 66 --out data/coh_n4-8.csv.gz
python scripts/make_kladder.py --base data/coh_n4-8.csv.gz --k-list 16 32 --per-k 8 --jobs $J --out data/coh_kladder_n4-8.csv.gz
python scripts/make_dataset.py --family tfim --coh-rx 0.05 --coh-zz 0.08 --n-list 4 6 8 --steps-list 4 8 12 \
    --targets-per-config 30 --nc-per-target 16 --shots 2000 --jobs $J --seed 77 --out data/coh2_n4-8.csv.gz
python scripts/make_kladder.py --base data/coh2_n4-8.csv.gz --k-list 16 32 --per-k 8 --jobs $J --out data/coh2_kladder_n4-8.csv.gz
