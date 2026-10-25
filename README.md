# OpinionLineage-OLS

Code and data for the manuscript

> R. R. Pathak, S. Vibhanshu, and Ranjith Kumar M, "Confidence-Weighted Opinion Lineage for Organic Sentiment Estimation Under Synthetic and Coordinated Amplification, With Exact Lineage-Bounded Retroactive Correction," submitted to *IEEE Access*, 2026.

The **Opinion Lineage System (OLS)** estimates *organic* sentiment, meaning the sentiment distribution of independently originating, authentic opinions, from streams that also contain reposts, quotations, paraphrases, coordinated copies and synthetic (AI-style) rewrites. When the provenance of stored records is revised later, OLS corrects an estimate it has already published. The correction is exact, and its cost is proportional to the affected lineage.

* **Opinion-Lineage Graph inferred from observable evidence only.** Candidate parents are the top-K earlier lexical neighbours. A calibrated gradient-boosted edge model combines semantic, containment, temporal, metadata, synthetic-text and coordination evidence into derivation confidences.
* **Ancestor-local contribution weights.** Each record's weight is `W_i = A_root * prod(1 - alpha c_e) * F_i`. Derivatives are retained rather than deleted, so both independent-opinion mass and amplification mass stay measurable.
* **Selective Dependency Rollback.** Per-record contribution witnesses and a stored aggregate `(A_c, Z)` turn any provenance revision into an additive delta over the descendant set D(v).
  * The result is provably identical to full recomputation.
  * The cost is O(|D(v)|), with zero model inference.
  * Confidence-weighted partial retraction is supported.
* **Error decomposition.** Total error splits into a classifier floor and a provenance error, the latter measured against a root oracle that uses the same classifier.

| Item | Location |
|---|---|
| Source code (this repository) | https://github.com/RakshitRPathak/OpinionLineage-OLS |
| Archived release + data (Zenodo) | https://doi.org/10.5281/zenodo.XXXXXXX *(fill in after the first release)* |
| Reproducible capsule (Code Ocean) | https://doi.org/10.24433/CO.XXXXXXX.v1 *(fill in after publishing the capsule)* |
| Licence | Code: MIT. Generated stream fields: CC BY 4.0. TweetEval: see `data/LICENSE-DATA.txt` |

## Layout (Code Ocean compatible)

```
code/
  run                  # Code Ocean entry point -> python main.py
  main.py              # reproduces every table and figure into ../results
  olq/
    data.py            # TweetEval loading, SHA-256 verification, fixed pool partition
    textops.py         # normalisation and derivative operators (repost, quote, paraphrase, synthetic rewrite, coordinated copy)
    stream.py          # contaminated-stream generator (hard negatives, metadata stripping, stealth bots, campaigns)
    models.py          # sentiment classifiers + selection, synthetic-text detector, coordination scorer
    lineage.py         # candidate retrieval, pair features, calibrated edge model, primary-parent forest, lineage metrics
    weights.py         # contribution weights, witness store, Selective Dependency Rollback (Kahan-compensated)
    quant.py           # weighted PCC / PACC / EM quantification
    pipeline.py        # end-to-end processing, baselines, ablations
    experiments.py     # experiments E1-E6
    figures.py         # manuscript figures
data/
  tweeteval/           # TweetEval sentiment split (unchanged copy) + SHA256SUMS
  streams/             # released canonical stream (kappa = 0.4, seed 0) with hidden ground-truth columns (true_*)
  LICENSE-DATA.txt
environment/           # pinned Python 3.13 environment (Dockerfile, requirements.txt)
results/               # written by the run (empty in the repository)
results_reference/     # outputs of the reference run reported in the paper
docs/fig_framework.*   # framework diagram (Graphviz source + render)
```

## Reproduce

```bash
pip install -r environment/requirements.txt
cd code
python main.py            # full run, about 30 min on 2 CPU cores
python main.py --quick    # smoke test, about 8 min
```

On Code Ocean, press **Reproducible Run**. `code/run` writes everything to `/results`. The run verifies the TweetEval files against `data/tweeteval/SHA256SUMS`. It also checks, by SHA-256, that the regenerated canonical stream matches `data/streams/stream_kappa0.4_seed0.csv`, and records the result in `results/summary.json`.

## Outputs

| File | Content |
|---|---|
| `table_sentiment_models.csv`, `table_synthetic_detector.csv` | component models (E1) |
| `table_quant_confusion_val_vs_test.csv` | evidence against the label-shift assumption |
| `e2_*` | contamination sweep: estimates, lineage metrics, SCI, independent opinion mass |
| `e3_robustness_per_seed.csv` | metadata exposure x hard negatives x paraphrase strength |
| `e4_ablation_per_seed.csv` | component ablation |
| `e5_*` | stealth-campaign discovery and retroactive correction policies |
| `e6_*` | rollback cost and exactness on a ~50k-record real-text stream, on synthetic forests up to 10^6 records, and numerical drift |
| `fig_*.png/pdf` | manuscript figures |

## Released stream schema

Columns used by the method:
* `id`, `t`, `day`, `account`, `text`, `meta_parent` (exposed repost/quote parent or -1)
* `acc_log_age`, `acc_log_ppd`, `acc_regularity`, `acc_default`

Hidden ground truth, never read by the estimators:
* `true_y`, `true_type`, `true_parent`, `true_root`, `true_authentic`, `true_hard_negative`, `true_family`, `true_campaign`, `true_bot`

## Citation

See `CITATION.cff`.
