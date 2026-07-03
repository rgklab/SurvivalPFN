<h1 align="center">SurvivalPFN</h1>
<p align="center"><b>Amortizing Survival Prediction via In-Context Bayesian Inference</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.15488"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.15488-b31b1b.svg"></a>
  <a href="https://huggingface.co/shi-ang/SurvivalPFN"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-yellow.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="#citation"><img alt="Cite" src="https://img.shields.io/badge/Cite-BibTeX-informational.svg"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.15488">Paper</a> &nbsp;·&nbsp;
  <a href="#install">Install</a> &nbsp;·&nbsp;
  <a href="#quick-inference">Quick Inference</a> &nbsp;·&nbsp;
  <a href="#dataset-evaluation">Evaluation</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#reproducing-the-results">Reproduce</a> &nbsp;·&nbsp;
  <a href="#citation">Cite</a>
</p>

---

**SurvivalPFN** is a prior-data fitted network (PFN) for right-censored survival analysis. It is described in the accompanying paper: **[Amortizing Survival Prediction via In-Context Bayesian Inference](https://arxiv.org/abs/2605.15488)**.

In survival analysis, each row has covariates, an observed time, and an event indicator. If the event indicator is 1, the observed time is the event time. If it is 0, the event has not yet been observed by that time, so the true event time is only known to be later. SurvivalPFN predicts a posterior predictive event-time distribution for new rows from an in-context training set of covariates, observed times, and censoring indicators.

SurvivalPFN is trained on synthetic right-censored datasets sampled from survival priors. At inference time it does **not** fine-tune on the target dataset: it consumes the target dataset's context rows in a single forward pass and returns survival distributions for query rows.

## Authors

Shi-ang Qi, Vahid Balazadeh, Michael Cooper, Russell Greiner, and Rahul G. Krishnan.

## Install

CPU-only environment:

```bash
conda env create -f env-cpu.yaml
conda activate survivalpfn_cpu
```

GPU/CUDA environment:

```bash
conda env create -f env.yaml
conda activate survivalpfn_env
```

or:

```bash
pip install -r requirements.txt
```

## Quick Inference

Run the release-checkpoint quickstart:

```bash
python examples/quickstart.py
```

If CUDA is available, the same example runs on GPU:

```bash
python examples/quickstart.py --device cuda:0
```

The quickstart builds a tiny censored dataset, loads the Hugging Face checkpoint, predicts event-time distributions for held-out rows, and reports the concordance index (C-index) against the held-out labels:

```python
import numpy as np
import torch
from survivalpfn import SurvivalEstimator, harrell_concordance_index

def make_tiny_censored_dataset(n_train=64, n_test=8, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
    risk = 0.8 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2]
    event_time = np.exp(1.0 - risk + rng.normal(scale=0.35, size=X.shape[0]))
    censor_time = rng.exponential(scale=3.0, size=X.shape[0]) + 0.05
    T = np.minimum(event_time, censor_time).astype(np.float32)
    delta = (event_time <= censor_time).astype(np.float32)
    tr, te = slice(0, n_train), slice(n_train, None)
    return X[tr], T[tr], delta[tr], X[te], T[te], delta[te]

X_train, T_train, delta_train, X_test, T_test, delta_test = make_tiny_censored_dataset()

estimator = SurvivalEstimator(device="cpu", model_path="shi-ang/SurvivalPFN")
estimator.fit(X=X_train, T=T_train, delta=delta_train)

dist = estimator.predict_event_distribution(X_test)
predicted_times = dist.median().detach().cpu().numpy()

c_index = harrell_concordance_index(preds=predicted_times, T=T_test, delta=delta_test)
print(f"concordance_index={c_index:.3f}")  # clears random (0.5) on this structured data
```

By default, `SurvivalEstimator` downloads `survivalpfn_v0.1.pt` from [`shi-ang/SurvivalPFN`](https://huggingface.co/shi-ang/SurvivalPFN) on Hugging Face and caches it under `~/.cache/survivalpfn`. To use a local checkpoint, pass its path instead:

```python
estimator = SurvivalEstimator(model_path="output/checkpoints/run/latest.pt", device="cuda:0")
```

## Dataset Evaluation

Evaluate SurvivalPFN on a single dataset:

```bash
python -m survivalpfn.inference.evaluate --data PBC --n-exp 10 --seed 0 --device cpu --out output/pbc_survivalpfn.csv
```

By default the evaluator downloads the Hugging Face release checkpoint; pass `--model-path output/checkpoints/run/latest.pt` to score a local checkpoint instead.

Two flags control how the numbers are computed:

- `--metrics-backend` selects the metric implementation. `paper` (default) uses the exact [`SurvivalEVAL`](https://github.com/shi-ang/SurvivalEVAL) routines behind our reported results; `lightweight` uses local `lifelines`-based metrics with fewer dependencies.
- `--preprocess true|false` toggles covariate imputation and scaling. It defaults to our evaluation protocol, which leaves covariates untouched for every dataset except MSKCC.

Run a one-split smoke test over the checkpoint-selection datasets:

```bash
python -m survivalpfn.inference.evaluate --data paper-validation --fixed-split true --n-exp 1 --seed 0 --device cuda:0 --out output/validation_nexp1.csv
```

Run the full held-out benchmark protocol:

```bash
python -m survivalpfn.inference.evaluate --data paper-test --n-exp 10 --seed 0 --device cuda:0 --out output/survivalpfn_test.csv
```

The dataset groups are defined in `survivalpfn.inference.paper` as `PAPER_VALIDATION_DATASETS` and `PAPER_TEST_DATASETS`. The full held-out run includes large datasets and can take considerable wall time, so run it on a large-memory GPU rather than as an installation check.

## Training

The default training config reproduces the released checkpoint:

- survival-distribution prior
- lognormal time transform
- NLL objective through `query_strategy=both`
- fixed 70/30 context-query split
- checkpoint selection by weighted average integrated Brier score

Train from scratch:

```bash
python train.py +experiment=default
```

Run a short local smoke test:

```bash
python train.py +experiment=simple trainer.max_epochs=1 trainer.num_model_updates=1 trainer.num_agg=1 trainer.batch_size=4 num_workers=0 compile=false default_device=cpu
```

Set `default_device=cuda:0` to train on one GPU or `default_device=cpu` for local CPU smoke tests.

Create a resumable local smoke checkpoint:

```bash
python train.py +experiment=simple +callbacks@callbacks.checkpoint=checkpoint callbacks.checkpoint.monitor=train_loss callbacks.checkpoint.checkpoint_dir_name=readme_resume_smoke trainer.max_epochs=1 trainer.num_model_updates=1 trainer.num_agg=1 trainer.batch_size=4 num_workers=0 compile=false default_device=cpu
```

Resume from that checkpoint:

```bash
python train.py +experiment=simple trainer.max_epochs=2 trainer.num_model_updates=1 trainer.num_agg=1 trainer.batch_size=4 num_workers=0 compile=false default_device=cpu resume_training=enabled resume_training.checkpoint_path=output/checkpoints/readme_resume_smoke/latest.pt
```

## Reproducing the Results

`results/paper/benchmark_results.csv` holds the reported held-out benchmark numbers, with the schema `model,dataset,metric,mean,std,runtime_seconds`. The public Hugging Face release checkpoint ([`shi-ang/SurvivalPFN`](https://huggingface.co/shi-ang/SurvivalPFN), `survivalpfn_v0.1.pt`) reproduces these numbers closely across the AIDS, BMT, METABRIC, PBC, and SUPPORT datasets.

Regenerate a fresh table by running the held-out benchmark command from [Dataset Evaluation](#dataset-evaluation), then plot any benchmark CSV offline:

```bash
python -m survivalpfn.inference.paper_results --results results/paper/benchmark_results.csv --outdir output/plots
```

Reference implementations of the comparison models are not included in this release; the benchmark plots are reproduced from the frozen result table above.

## Layout

```text
survivalpfn/
  survival_estimator.py      Public inference estimator
  evaluation.py              Metric helpers
  models/                    In-context transformer model and checkpoint loading
  prior/                     Synthetic survival priors and TabDPT-style table generation
  callbacks/                 Training callbacks
  inference/
    data/                    Dataset loaders
    evaluate.py              SurvivalPFN dataset evaluation CLI
    paper.py                 Benchmark dataset groups and result schemas
    paper_results.py         Offline plotting for benchmark tables
conf/                        Hydra training configuration
results/paper/               Frozen benchmark result table
tests/                       Unit and smoke tests
```

## Tests

```bash
pytest -q
```

Useful focused checks:

```bash
pytest -q tests/test_datasets.py
pytest --integration -k "training or priors or time_transform"
python examples/quickstart.py
python -m survivalpfn.inference.paper_results --results results/paper/benchmark_results.csv --outdir output/plots
python train.py +experiment=simple trainer.max_epochs=1 trainer.num_model_updates=1 trainer.num_agg=1 trainer.batch_size=4 num_workers=0 compile=false default_device=cpu
```

## Citation

If you use this work, please cite:

```bibtex
@article{qi2026survivalpfn,
  title         = {SurvivalPFN: Amortizing Survival Prediction via In-Context Bayesian Inference},
  author        = {Qi, Shi-ang and Balazadeh, Vahid and Cooper, Michael and Greiner, Russell and Krishnan, Rahul G.},
  journal       = {arXiv preprint arXiv:2605.15488},
  year          = {2026},
  eprint        = {2605.15488},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
