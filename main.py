"""Reproduce every table and figure of the manuscript into ../results (Code Ocean: /results).

    python main.py            # full run (~30 min on 2 CPUs)
    python main.py --quick    # smoke test with fewer seeds and smaller sweeps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olq import data, experiments as ex, figures, stream as st  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RELEASED = os.path.join(DATA, "streams", "stream_kappa0.4_seed0.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.environ.get("RESULTS", os.path.join(HERE, "..", "results")))
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    seeds = [0, 1] if a.quick else [0, 1, 2, 3, 4]
    kappas = [0.0, 0.3, 0.6] if a.quick else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    pools = data.load_tweeteval(os.path.join(DATA, "tweeteval"))
    stats = pd.DataFrame([dict(pool=k, n=len(v), **{f"share_{c}": float((v.y == i).mean()) for i, c in enumerate(data.LABELS)})
                          for k, v in pools.items()])
    stats.to_csv(f"{out}/table_data_pools.csv", index=False)
    log("E1 fitting components (sentiment suite, synthetic detector, calibration streams, edge model, tuning)")
    comp = ex.fit_components(pools, seed=0, log=log)
    X = comp.extra
    X["sent_tab"].to_csv(f"{out}/table_sentiment_models.csv", index=False)
    pd.DataFrame(X["cm"], index=data.LABELS, columns=data.LABELS).to_csv(f"{out}/table_confusion_test.csv")
    X["syn_tab"].to_csv(f"{out}/table_synthetic_detector.csv", index=False)
    X["grid"].to_csv(f"{out}/table_weight_tuning_grid.csv", index=False)
    X["nd_grid"].to_csv(f"{out}/table_neardup_tuning.csv", index=False)
    from olq import quant as qt
    Mt = qt.confusion(X["P_test"], X["y_test"])
    pd.DataFrame(np.c_[comp.M, Mt], index=data.LABELS,
                 columns=[f"val_{c}" for c in data.LABELS] + [f"test_{c}" for c in data.LABELS]).to_csv(f"{out}/table_quant_confusion_val_vs_test.csv")

    # released canonical stream + reproducibility checksum
    df0 = ex.make_stream(pools, comp, st.StreamConfig(kappa=0.4), 0)
    rel = df0.drop(columns=["src"])
    os.makedirs(os.path.dirname(RELEASED), exist_ok=True)
    tmp = f"{out}/stream_kappa0.4_seed0.csv"
    rel.to_csv(tmp, index=False, float_format="%.6f")
    h_new = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    h_ref = hashlib.sha256(open(RELEASED, "rb").read()).hexdigest() if os.path.exists(RELEASED) else None
    log(f"released stream sha256 {h_new[:16]}... matches shipped copy: {h_ref == h_new if h_ref else 'n/a (no shipped copy)'}")

    log("E2 contamination sweep")
    R, G, SCI, IOM = ex.e2_contamination(pools, comp, seeds, kappas, log=log)
    R.to_csv(f"{out}/e2_contamination_per_seed.csv", index=False); G.to_csv(f"{out}/e2_lineage_per_seed.csv", index=False)
    SCI.to_csv(f"{out}/e2_sci_per_seed.csv", index=False); IOM.to_csv(f"{out}/e2_iom_per_seed.csv", index=False)
    log("E3 robustness grid")
    RB = ex.e3_robustness(pools, comp, seeds[:3] if not a.quick else seeds[:1], log=log)
    RB.to_csv(f"{out}/e3_robustness_per_seed.csv", index=False)
    log("E4 ablations")
    AB = ex.e4_ablation(pools, comp, seeds, log=log)
    AB.to_csv(f"{out}/e4_ablation_per_seed.csv", index=False)
    log("E5 stealth-campaign discovery and retroactive correction")
    CP, series = ex.e5_campaign(pools, comp, seeds, log=log)
    CP.to_csv(f"{out}/e5_campaign_per_seed.csv", index=False); series.to_csv(f"{out}/e5_daily_series_seed0.csv", index=False)
    log("E6 rollback at scale")
    RS, dfL, obsL, gL = ex.e6_real_scale(pools, comp, n_roots=2500 if a.quick else 10_000, log=log)
    RS.to_csv(f"{out}/e6_real_scale.csv", index=False)
    SC = ex.e6_rollback_scaling([10_000, 100_000] if a.quick else [10_000, 100_000, 1_000_000],
                                float(RS.inference_s_per_record.iat[0]), log=log)
    SC.to_csv(f"{out}/e6_scaling.csv", index=False)
    DR = ex.e6_drift(n=20_000 if a.quick else 100_000, steps=10_000 if a.quick else 100_000,
                     every=1_000 if a.quick else 5_000, log=log)
    DR.to_csv(f"{out}/e6_drift.csv", index=False)

    log("figures")
    figures.make_all(out, dict(R=R, G=G, SCI=SCI, IOM=IOM, RB=RB, AB=AB, CP=CP, series=series, SC=SC, DR=DR, RS=RS,
                               cm=X["cm"], example=(df0, comp), alpha=comp.abg[0]))
    summary = dict(selected_sentiment_model=X["sent_best"], abg=list(map(float, comp.abg)), tau_nd=X["tau_nd"],
                   cal_edge_auc=float(X["cal_edge_auc"]), seeds=seeds, kappas=kappas, stream_sha256=h_new,
                   stream_matches_shipped=(h_ref == h_new) if h_ref else None, runtime_s=time.time() - t0, quick=a.quick)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    log(f"done: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
