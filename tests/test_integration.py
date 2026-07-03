import os
import shutil
import subprocess

import pytest
import torch


def _small_training_overrides(device: str = "cpu") -> list[str]:
    return [
        "trainer.max_epochs=1",
        "trainer.num_agg=1",
        "trainer.num_model_updates=1",
        "trainer.batch_size=2",
        "trainer.grad_clip=null",
        "compile=false",
        "num_workers=0",
        f"default_device={device}",
        "model.obj.model.nbins=64",
        "model.obj.model.nhead=2",
        "model.obj.model.nhid=32",
        "model.obj.model.ninp=16",
        "model.obj.model.nlayers=1",
        "model.obj.model.num_features=16",
    ]


def _small_direct_prior_overrides() -> list[str]:
    return [
        "train_meta_dataset.n_samples=128",
        "train_meta_dataset.max_n_covariates=15",
        "train_meta_dataset.post_padding_n_cols=15",
    ]


def _small_kitchen_sink_overrides() -> list[str]:
    overrides = [
        "train_meta_dataset.priors.d1.n_samples=128",
        "train_meta_dataset.priors.d1.max_n_covariates=15",
        "train_meta_dataset.priors.d1.post_padding_n_cols=15",
        "train_meta_dataset.priors.d2.n_samples=128",
        "train_meta_dataset.priors.d2.max_n_covariates=15",
        "train_meta_dataset.priors.d2.post_padding_n_cols=15",
        "train_meta_dataset.priors.d3.n_samples=128",
        "train_meta_dataset.priors.d3.max_n_covariates=15",
        "train_meta_dataset.priors.d3.post_padding_n_cols=15",
    ]
    return overrides


def _train_cmd(*overrides: str, device: str = "cpu") -> list[str]:
    return [
        "python",
        "train.py",
        "+experiment=simple",
        *_small_training_overrides(device=device),
        *overrides,
    ]


def _run_subprocess(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=".",
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f">> Failed!\n"
            f"[COMMAND]: {' '.join(cmd)}\n"
            f"[EXIT CODE]: {e.returncode}\n"
            f"[STDOUT]:\n{e.stdout}\n"
            f"[STDERR]:\n{e.stderr}"
        )


PRIOR_COMMANDS = [
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
        ),
        id="survival_dist_prior",
    ),
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=mixture_prior",
            *_small_direct_prior_overrides(),
            "+train_meta_dataset.dist=LogNormal",
        ),
        id="mixture_prior_lognormal",
    ),
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=mixture_prior",
            *_small_direct_prior_overrides(),
            "+train_meta_dataset.dist=Weibull",
        ),
        id="mixture_prior_weibull",
    ),
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=kitchen_sink_prior",
            *_small_kitchen_sink_overrides(),
        ),
        id="kitchen_sink_prior",
    ),
]

TIME_TRANSFORM_COMMANDS = [
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
            "time_transform=lognormal",
        ),
        id="lognormal",
    ),
    pytest.param(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
            "time_transform=quantile",
        ),
        id="quantile",
    ),
]


@pytest.mark.integration
def test_training_checkpoint_resume_smoke_cpu():
    checkpoint_dir = "output/checkpoints/integration_test"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    _run_subprocess(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
            "+callbacks@callbacks.checkpoint=checkpoint",
            "callbacks.checkpoint.frequency=1",
            "callbacks.checkpoint.checkpoint_dir_name=integration_test",
        )
    )
    _run_subprocess(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
            "resume_training=enabled",
            "trainer.max_epochs=2",
            'resume_training.checkpoint_path="output/checkpoints/integration_test/latest.pt"',
        )
    )
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.parametrize("cmd", PRIOR_COMMANDS)
def test_priors_training_smoke_cpu(cmd: list[str]):
    _run_subprocess(cmd)


@pytest.mark.integration
@pytest.mark.parametrize("cmd", TIME_TRANSFORM_COMMANDS)
def test_time_transform_training_smoke_cpu(cmd: list[str]):
    _run_subprocess(cmd)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_training_smoke_cuda():
    _run_subprocess(
        _train_cmd(
            "meta_dataset@train_meta_dataset=survival_dist_prior",
            *_small_direct_prior_overrides(),
            device="cuda:0",
        )
    )
