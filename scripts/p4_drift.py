"""Phase 4 (RQ3): keeping coverage of mitigated values while the device noise drifts.

  python scripts/p4_drift.py --base data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz \
      --drift data/drift_n4-8.csv.gz --seeds 10

Setup (per seed): base-dataset groups with steps = 8 are split 50/50 into train / calibration.
vnCDR (pooled linear, per (n, steps, obs_weight) cell) is trained once on near-Clifford circuits
(k <= 8) at t = 0 and never retrained. The k-scale model g(k) and the initial normalized
calibration scores come from the calibration groups (k-scaled CP, RQ2). Each n is its own stream.

At every time t the policies give an interval for the TARGET circuits at t using only probe
circuits (near-Clifford, exact labels) observed before t:
  static     initial calibration, never updated
  periodic   every 50 steps: re-calibrate on the probes of the last 50 steps
  window     every step: calibrate on the probes of the last 50 steps
  aci        adaptive conformal inference (Gibbs & Candes 2021) on the initial scores, gamma = 0.005
  aci_mart   aci + conformal test martingale on probe p-values (mixture of power martingales);
             alarm when the martingale >= 1/delta (delta = 0.01, Ville). After an alarm the next 20
             probes become the new reference set, then alpha and the martingale are reset.
Metrics per scenario: coverage (all t, t >= 150), worst 50-step rolling coverage, mean width,
alarms (before / after t = 150) and detection delay (first alarm at or after t = 150).
Outputs: results/p4_summary.csv, results/p4_rolling.csv, results/p4_rolling.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from g0_split_cp import LinearCell, conformal_q
from p3_weighted_cp import kscale_fit

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["static", "periodic", "window", "aci", "aci_mart", "refit", "track"]
CHANGE_T = 150
EPS_GRID = np.linspace(0.05, 0.95, 19)


def calibrate(base: pd.DataFrame, n_list, seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    b = base[(base.steps == 8) & base.n.isin(n_list)]
    tg = b[b.kind == "target"].drop_duplicates("group_id")
    train_ids, cal_a, cal_b = set(), set(), set()
    for _, g in tg.groupby("n"):
        ids = rng.permutation(g.group_id.values)
        h = len(ids) // 2
        train_ids |= set(ids[:h])
        rest = ids[h:]
        cal_a |= set(rest[: len(rest) // 2])
        cal_b |= set(rest[len(rest) // 2:])
    nc = b[b.kind == "nearclifford"]
    model = LinearCell().fit(nc[nc.group_id.isin(train_ids) & (nc.n_keep <= 8)])
    ca = nc[nc.group_id.isin(cal_a)]
    g, _ = kscale_fit(ca, ca.y_ideal.to_numpy() - model.predict(ca))
    cb = nc[nc.group_id.isin(cal_b)]
    s = np.abs(cb.y_ideal.to_numpy() - model.predict(cb)) / g(cb)
    ref = {}
    for n in n_list:
        m = (cb.n == n).to_numpy()
        circ = pd.Series(s[m]).groupby(cb.circuit_id.to_numpy()[m]).mean().to_numpy()
        ref[n] = (s[m], circ)  # row scores (for quantiles), per-circuit mean scores (for p-values)
    return model, g, ref


def p_value(score, ref_circ, rng):
    return (np.sum(ref_circ > score) + rng.uniform() * (np.sum(ref_circ == score) + 1)) / (len(ref_circ) + 1)


def run_stream(stream: pd.DataFrame, model, g, ref, alpha, gamma, delta, rng, track_w=50):
    """stream: rows of one (scenario, n), all t. Returns per-target-row results for every policy."""
    stream = stream.assign(pred=model.predict(stream), gk=g(stream))
    stream["score"] = np.abs(stream.y_ideal - stream.pred) / stream.gk
    T = int(stream.t.max()) + 1
    probes = [stream[(stream.t == t) & (stream.role == "probe")] for t in range(T)]
    targets = [stream[(stream.t == t) & (stream.role != "probe")] for t in range(T)]
    s0_rows, s0_circ = ref
    q0 = conformal_q(s0_rows, alpha)

    out = []
    st = {"periodic": q0, "refit_m": model, "refit_q": q0, "aci_a": alpha, "am_a": alpha, "am_ref": s0_rows, "am_refc": s0_circ,
          "logm": np.zeros_like(EPS_GRID), "collect": None, "alarms": []}
    hist = []  # probe row scores per t
    for t in range(T):
        recent = np.concatenate(hist[max(0, t - 50):t]) if t > 0 else np.array([])
        if t > 0 and t % 50 == 0:
            st["periodic"] = conformal_q(recent, alpha)
            win = pd.concat(probes[max(0, t - 100):t])
            tr, ca = win[win.t % 2 == 0], win[win.t % 2 == 1]
            try:
                m_new = LinearCell().fit(tr)
                st["refit_q"] = conformal_q(np.abs(ca.y_ideal.to_numpy() - m_new.predict(ca)) / ca.gk.to_numpy(), alpha)
                st["refit_m"] = m_new
            except (KeyError, np.linalg.LinAlgError):
                pass
        if t == 50:
            st["probe_q_ref"] = conformal_q(recent, alpha)  # probes of the first 50 (stable) steps
        if t < 50:
            track_q = q0
        else:
            rec_w = np.concatenate(hist[t - track_w:t])
            track_q = q0 * conformal_q(rec_w, alpha) / st["probe_q_ref"]
        q = {
            "track": track_q,
            "static": q0,
            "periodic": st["periodic"],
            "window": conformal_q(recent, alpha) if len(hist[max(0, t - 50):t]) >= 20 else q0,
            "aci": np.inf if st["aci_a"] <= 0 else (0.0 if st["aci_a"] >= 1 else np.quantile(s0_rows, 1 - st["aci_a"], method="higher")),
            "aci_mart": np.inf if st["am_a"] <= 0 else (0.0 if st["am_a"] >= 1 else np.quantile(st["am_ref"], 1 - st["am_a"], method="higher")),
        }
        tg = targets[t]
        for name, qq in q.items():
            out.append(pd.DataFrame({"t": t, "policy": name, "covered": (tg.score <= qq).to_numpy(),
                                     "width": 2 * qq * tg.gk.to_numpy()}))
        sc_refit = np.abs(tg.y_ideal.to_numpy() - st["refit_m"].predict(tg)) / tg.gk.to_numpy()
        out.append(pd.DataFrame({"t": t, "policy": "refit", "covered": sc_refit <= st["refit_q"],
                                 "width": 2 * st["refit_q"] * tg.gk.to_numpy(),
                                 "abs_err": np.abs(tg.y_ideal.to_numpy() - st["refit_m"].predict(tg))}))
        out[-2] = out[-2]  # keep order
        out.append(pd.DataFrame({"t": t, "policy": "_base_err", "covered": np.nan, "width": np.nan,
                                 "abs_err": np.abs(tg.y_ideal - tg.pred).to_numpy()}))
        # observe probes at t
        pr = probes[t]
        ps = pr.score.to_numpy()
        hist.append(ps)
        st["aci_a"] += gamma * (alpha - np.mean(ps > q["aci"]))
        st["am_a"] += gamma * (alpha - np.mean(ps > q["aci_mart"]))
        if st["collect"] is not None:  # gathering a new reference set after an alarm
            st["collect"].append(pr)
            if len(st["collect"]) >= 20:
                c = pd.concat(st["collect"])
                st["am_ref"] = c.score.to_numpy()
                st["am_refc"] = c.groupby("circuit_id").score.mean().to_numpy()
                st["am_a"], st["logm"], st["collect"] = alpha, np.zeros_like(EPS_GRID), None
        else:
            p = p_value(ps.mean(), st["am_refc"], rng)
            st["logm"] += np.log(EPS_GRID) + (EPS_GRID - 1) * np.log(max(p, 1e-12))
            if np.log(np.mean(np.exp(st["logm"]))) >= np.log(1 / delta):
                st["alarms"].append(t)
                st["collect"] = []
    return pd.concat(out), st["alarms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=[str(ROOT / "data" / "main_n4-8.csv.gz"), str(ROOT / "data" / "kladder_n4-8.csv.gz")])
    ap.add_argument("--drift", default=str(ROOT / "data" / "drift_n4-8.csv.gz"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--track-window", type=int, default=10, help="probe window (steps) of the track policy")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    base = pd.concat([pd.read_csv(p) for p in args.base], ignore_index=True)
    drift = pd.read_csv(args.drift)
    n_list = sorted(drift.n.unique())

    res, alarms = [], []
    for seed in range(args.seeds):
        model, g, ref = calibrate(base, n_list, seed, args.alpha)
        rng = np.random.default_rng(100 + seed)
        for (scen, n), stream in drift.groupby(["scenario", "n"]):
            r, al = run_stream(stream, model, g, ref[n], args.alpha, args.gamma, args.delta, rng, args.track_window)
            res.append(r.assign(seed=seed, scenario=scen, n=n))
            alarms.append(dict(seed=seed, scenario=scen, n=n, alarms=al,
                               before=sum(a < CHANGE_T for a in al),
                               delay=next((a - CHANGE_T for a in al if a >= CHANGE_T), np.nan)))
        print(f"  seed {seed + 1}/{args.seeds}", flush=True)
    res = pd.concat(res)
    mae = res[res.policy.isin(["_base_err", "refit"])].groupby(["scenario", "policy"]).abs_err.mean().unstack()
    res = res[res.policy != "_base_err"]
    al = pd.DataFrame(alarms)

    # rolling coverage (50 steps) per seed/scenario/policy, pooled over n
    per_t = res.groupby(["seed", "scenario", "policy", "t"]).covered.mean().reset_index()
    per_t["roll"] = per_t.groupby(["seed", "scenario", "policy"]).covered.transform(lambda x: x.rolling(50, min_periods=50).mean())
    worst = per_t.groupby(["seed", "scenario", "policy"]).roll.min().groupby(["scenario", "policy"]).mean()
    summ = res.groupby(["scenario", "policy"]).agg(coverage=("covered", "mean"),
                                                   width=("width", lambda w: np.mean(np.minimum(w, 2.0))))
    summ["cov_after_150"] = res[res.t >= CHANGE_T].groupby(["scenario", "policy"]).covered.mean()
    summ["worst_roll50"] = worst
    am = al.groupby("scenario").agg(alarms_before_150=("before", "mean"),
                                    detect_rate=("delay", lambda x: np.mean(np.isfinite(x))),
                                    median_delay=("delay", "median"))
    pd.set_option("display.width", 200)
    print(f"\nalpha = {args.alpha}, gamma = {args.gamma}, delta = {args.delta}, {args.seeds} seeds, streams n = {n_list}\n")
    for scen, s in summ.groupby(level=0):
        print(f"--- {scen}")
        print(s.droplevel(0).reindex(POLICIES).round(3).to_string())
    print("\nMAE of mitigated targets: frozen vnCDR (_base_err) vs refit on probes:")
    print(mae.round(4).to_string())
    print("\nMartingale alarms (per stream, mean over seeds and n):")
    print(am.round(2).to_string())
    summ.to_csv(out / "p4_summary.csv")
    am.to_csv(out / "p4_alarms.csv")
    roll = per_t.groupby(["scenario", "policy", "t"]).roll.mean().reset_index()
    roll.to_csv(out / "p4_rolling.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_pols = {"track": ("#257de0", "-", "track (proposed)"), "aci": ("#f47230", "-", "ACI"),
                     "window": ("#00b894", "-", "sliding window"), "static": ("#8c8c8c", "--", "static")}
        titles = {"S0": "S0 control", "S2": "S2 gradual drift", "S3": "S3 jump at t = 150",
                  "S4": "S4 random walk", "S5": "S5 device swap at t = 150"}
        scens = sorted(roll.scenario.unique())
        fig, axes = plt.subplots(1, len(scens), figsize=(3.2 * len(scens), 3.2), sharey=True)
        for ax, sc in zip(np.atleast_1d(axes), scens):
            for pol, (col, ls, lab) in plot_pols.items():
                r = roll[(roll.scenario == sc) & (roll.policy == pol)]
                ax.plot(r.t, r.roll, color=col, lw=2, ls=ls, label=lab)
            ax.axhline(1 - args.alpha, color="#191919", lw=0.8, ls=":")
            if sc in ("S3", "S5"):
                ax.axvline(CHANGE_T, color="#bbbbbb", lw=0.8)
            ax.set_title(titles.get(sc, sc), loc="left", fontsize=10, fontweight="bold")
            ax.set_xlabel("time step", fontsize=9, color="#666666")
            ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel("50-step rolling coverage", fontsize=9)
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="lower left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"p4_rolling.{ext}", dpi=200)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
