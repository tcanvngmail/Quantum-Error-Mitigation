#!/bin/bash
# Reproduce every table and figure of the paper from the released data in data/.
# No quantum simulation is needed. Logs go to results/logs/, tables and figures to results/.
set -e
export OMP_NUM_THREADS=1
mkdir -p results/logs
D=data
MAIN="$D/main_n4-8.csv.gz $D/kladder_n4-8.csv.gz $D/main_n10.csv.gz $D/kladder_n10.csv.gz"
XXZ="$D/xxz_n4-8.csv.gz $D/xxz_kladder_n4-8.csv.gz"
COHA="$D/coh_n4-8.csv.gz $D/coh_kladder_n4-8.csv.gz"
COHB="$D/coh2_n4-8.csv.gz $D/coh2_kladder_n4-8.csv.gz"

echo "[1/7] Sec. 6.1: split CP baseline (G0)"
python scripts/g0_split_cp.py --data $D/main_n4-8.csv.gz --reps 50 > results/logs/g0.log

echo "[2/7] Sec. 6.2, Table 2, Figs. 2, 3, B1: circuit shift"
python scripts/p3_weighted_cp.py --data $MAIN --reps 50 --model linear --dump > results/logs/p3_main_linear.log
python scripts/p3_weighted_cp.py --data $MAIN --reps 50 --model gbm --dump > results/logs/p3_main_gbm.log
python scripts/fig_rq2.py --data $MAIN

echo "[3/7] Sec. 6.3, Tables 3 and 4: existing error bars and alpha sensitivity"
python scripts/p6_baselines.py --data $MAIN --reps 30 --alphas 0.05 0.1 0.2 --out results/p6_main > results/logs/p6_main.log
for a in 0.05 0.2; do
  python scripts/p3_weighted_cp.py --data $MAIN --reps 30 --model gbm --alpha $a --out results/p3_gbm_a$a > results/logs/p3_gbm_a$a.log
done

echo "[4/7] Sec. 6.4, Table 5, Fig. 4: held-out XXZ and coherent noise"
for name in XXZ COHA COHB; do
  files=${!name}
  for m in linear gbm; do
    python scripts/p3_weighted_cp.py --data $files --reps 30 --model $m --out results/p3_${name}_$m > results/logs/p3_${name}_$m.log
  done
  python scripts/p6_baselines.py --data $files --reps 30 --alphas 0.1 --out results/p6_$name > results/logs/p6_$name.log
done
for name in MAIN XXZ COHA COHB; do
  python scripts/a2_diagnostic.py --data ${!name} > results/logs/a2_$name.log
done
python scripts/fig_heldout.py --out results/fig_heldout.pdf

echo "[5/7] Appendix C: pre-registered ladder variant"
for name in MAIN XXZ COHA COHB; do
  for m in linear gbm; do
    python scripts/p7_ladder.py --data ${!name} --model $m --out results/p7_${name}_$m > results/logs/p7_${name}_$m.log
  done
done

echo "[6/7] Sec. 6.5, Table 6, Fig. 5: noise drift"
python scripts/p4_drift.py --base $D/main_n4-8.csv.gz $D/kladder_n4-8.csv.gz --drift $D/drift_n4-8.csv.gz --seeds 10 > results/logs/p4.log

echo "[7/7] Sec. 6.6, Fig. 6: shot budget"
python scripts/p5_shots.py --data $MAIN --reps 30 > results/logs/p5.log

echo "Done. See results/ and results/logs/."
