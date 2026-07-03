import math
from typing import Any, Callable, Dict, List, Optional

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from survivalpfn.callbacks import CallbackContext, Checkpoint, EvalSurvival
from survivalpfn.models import InContextModel
from survivalpfn.prior.meta_dataset import MetaDataset
from survivalpfn.utils import (
    prescan_data_split_buckets,
    sample_data_split,
    seed_everything,
    validate_split_strategy,
)

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


def calculate_loss(
    model: InContextModel,
    device: str,
    batch: Dict[
        str, torch.Tensor
    ],  # batch should contain "X", "T", "delta", and "E". This is a SET of datasets.
    train_data_split: float,
) -> torch.Tensor:
    """
    This function uses an in-context model to compute the loss for training the survival (foundation) model.
    `model` itself is a PFN-style model that takes in `X_train` `delta_train, T_train` and an `X_query`
    and then produces event time distributions.

    *Note*:
        The final loss is the average of the table loss values over the batch of tables; however, if for any reason the this
        function determines that the loss is not valid (e.g. NaN or Inf), the loss is ignored on that table.

    Args:
        model: The in-context model to use for training.
        device: The device to use for training.
        batch: The batch of data to use for training.
        train_data_split: The fraction of training data to use for context for the current epoch.

    Returns:
        avg_total_loss: The average total loss for the batch.
    """
    # Covariates (X, T, delta)
    # shuffle the covariate columns to induce column permutation invariance
    X: torch.Tensor = batch["X"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows, num_features)
    idx = torch.randperm(X.shape[-1], device=device)  # create idx on GPU
    X = X[:, :, idx]  # shape: (batch_size, num_rows, num_features)

    # split into context and query sets
    num_rows = X.shape[1]
    split_pos = int(train_data_split * num_rows)
    split_pos = max(1, min(split_pos, num_rows - 1))  # Ensure valid range

    T: torch.Tensor = batch["T"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows)
    delta: torch.Tensor = batch["delta"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows)

    # Labels (E, C)
    E: torch.Tensor = batch["E"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows)
    C: torch.Tensor = batch["C"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows)

    # compute the cross-entropy loss for the event times. We can do this in the synthetic data setting because we know
    # the ground-truth event times.
    survival_losses = model(
        X_context=X[:, :split_pos],
        delta_context=delta[:, :split_pos],
        T_context=T[:, :split_pos],
        X_query=X[:, split_pos:],
        E_query=E[:, split_pos:],
        C_query=C[:, split_pos:],
    )

    # Use functional masking to avoid data-dependent control flow for torch.compile
    # Compute validity mask for each sample
    valid_mask = torch.ones(batch["X"].shape[0], dtype=torch.bool, device=device)
    valid_mask &= ~torch.isnan(survival_losses)
    valid_mask &= ~torch.isinf(survival_losses)

    # Use functional approach: set invalid losses to 0 and normalize by valid count
    # This avoids the data-dependent 'if not torch.any(valid_mask)' branch
    valid_losses = torch.where(
        valid_mask, survival_losses, torch.zeros_like(survival_losses)
    )
    num_valid = valid_mask.sum().clamp(min=1)  # Clamp to avoid division by zero
    avg_loss = valid_losses.sum() / num_valid

    return avg_loss


def _handle_gradient_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float],
    device: str,
) -> tuple[bool, float]:
    """
    Handle gradient clipping and validation.

    Args:
        model: The model being trained
        optimizer: The optimizer
        grad_clip: Maximum gradient norm (None for no clipping)
        device: Device being used

    Returns:
        Tuple of (should_skip, grad_norm) where should_skip indicates if this step should be skipped
    """
    # Clip gradients and get norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=grad_clip or float("inf")
    ).item()

    # Check if gradients are valid
    skip_local = (grad_norm == 0) or math.isinf(grad_norm) or math.isnan(grad_norm)
    skip_flag = torch.tensor(skip_local, device=device, dtype=torch.uint8)

    should_skip = skip_flag.item()

    if should_skip:
        optimizer.zero_grad(set_to_none=True)
        print("[Warning] non-finite, NaN, or empty gradients – skipping update.")

    return should_skip, grad_norm


def _log_training_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_norm: float,
) -> None:
    """
    Log training metrics to wandb.

    Args:
        model: The model being trained
        optimizer: The optimizer
        grad_norm: Current gradient norm
    """
    with torch.no_grad():
        weight_norm = torch.linalg.vector_norm(
            torch.stack([p.norm() for p in model.parameters()])
        ).item()

    avg_lr = sum(
        [optimizer.param_groups[i]["lr"] for i in range(len(optimizer.param_groups))]
    ) / len(optimizer.param_groups)

    wandb.log(
        {
            "weights/grad_norm": grad_norm,
            "weights/weight_norm": weight_norm,
            "weights/avg_lr": avg_lr,
        }
    )


def _order_callbacks(callbacks: List[Any]) -> List[Any]:
    needs_eval_first = any(
        isinstance(callback, Checkpoint) and callback.requires_eval
        for callback in callbacks
    )
    if not needs_eval_first:
        return callbacks
    return sorted(
        callbacks, key=lambda callback: 0 if isinstance(callback, EvalSurvival) else 1
    )


def train(
    model: InContextModel,
    max_epochs: int,
    num_agg: int,  # number of forward passes to reach an effective batch size (e.g., if GPU can only fit 32 but you want 64, you do 2 aggregation steps)
    num_model_updates: int,  # number of model update steps per epoch
    optimizer_partial: Callable[[Any], torch.optim.Optimizer],
    train_meta_dataset: MetaDataset,
    callbacks: List[Any],
    lr_scheduler_partial: Optional[Callable[[torch.optim.Optimizer], Any]],
    compile: bool,
    num_workers: int,
    prefetch_factor: int,
    checkpoint: Optional[Dict[str, Any]],
    wandb_enabled: bool,
    batch_size: int,
    grad_clip: Optional[float],
    device: str,
    min_train_data_split: float,
    max_train_data_split: float,
    n_buckets: int,
) -> None:
    """
    Takes a model designed for prior-fitting (`model`) and then trains event times  on a given set of datasets.

    Evaluation is handled through callbacks. For a full list of callbacks, check the survivalpfn.callbacks
    package.
    """

    if checkpoint:  # load model state if resume = True
        model = InContextModel.load(
            model_state=checkpoint["model_state_dict"],
            model_config=checkpoint["model_config"],
        )
    effective_query_strategy = validate_split_strategy(
        model.query_strategy,
        min_train_data_split,
        max_train_data_split,
        n_buckets,
        warn=True,
    )
    if effective_query_strategy != model.query_strategy:
        model.query_strategy = effective_query_strategy
        model.model_config["query_strategy"] = effective_query_strategy

    model.to(device)
    model.train()

    if compile:
        model = torch.compile(model, dynamic=True)
    optimizer = optimizer_partial(model.get_param_groups())

    if checkpoint:  # load optimizer state if resume = True
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
        lr_scheduler_partial(optimizer) if lr_scheduler_partial is not None else None
    )
    data_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        data_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = torch.utils.data.DataLoader(train_meta_dataset, **data_loader_kwargs)
    train_loader_iterator = iter(train_loader)

    # create three progress bars
    pbar_train = tqdm(
        range(max_epochs * num_model_updates * num_agg),
        desc="Train Batches",
    )

    print(f"Effective batch size is: {num_agg * batch_size}")
    train_data_split_buckets = prescan_data_split_buckets(
        min_train_data_split,
        max_train_data_split,
        num_buckets=n_buckets,
    )
    print(
        "Using train/query split buckets: "
        + ", ".join(f"{split:.4f}" for split in train_data_split_buckets)
    )

    for epoch in range(max_epochs):
        if epoch < start_epoch:
            pbar_train.update(num_model_updates * num_agg)
            continue

        model.train()
        if hasattr(optimizer, "train"):  # for schedulefree
            optimizer.train()
        train_data_split = sample_data_split(train_data_split_buckets)

        # run the model for num_agg * num_model_updates iterations
        total_loss = 0.0
        for batch_counter in range(num_agg * num_model_updates):
            # accumate the loss gradients over num_agg batches for larger effective batch size
            train_batch = next(train_loader_iterator)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
            ):
                loss = calculate_loss(
                    model,
                    device,
                    batch=train_batch,
                    train_data_split=train_data_split,
                )
                loss = loss / num_agg

                total_loss += loss.item()
            last_micro_batch = ((batch_counter + 1) % num_agg) == 0
            loss.backward()

            if last_micro_batch:
                # Handle gradient clipping and validation
                should_skip, grad_norm = _handle_gradient_step(
                    model=model,
                    optimizer=optimizer,
                    grad_clip=grad_clip,
                    device=device,
                )

                pbar_train.update(num_agg)

                if should_skip:
                    continue

                # Log training metrics
                if wandb_enabled:
                    _log_training_metrics(
                        model=model, optimizer=optimizer, grad_norm=grad_norm
                    )

                # Update weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()

        # update the loss reports
        total_loss /= num_model_updates
        loss_report = {"train/loss": total_loss}

        # log the loss values
        pbar_train.set_postfix(loss_report)
        if wandb_enabled:
            wandb.log(loss_report)

        # make the optimizer eval mode
        if hasattr(optimizer, "eval"):
            optimizer.eval()

        callback_metrics: Dict[str, float] = {}

        # run the __call__ method of each callback
        for i, callback in enumerate(callbacks):
            pbar_train.set_description(
                f"Total Epochs (Callback [{i + 1}/{len(callbacks)}])"
            )

            # Create callback context
            context = CallbackContext(
                callback_idx=i,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=total_loss,
                device=device,
                wandb_enabled=wandb_enabled,
                callback_metrics=callback_metrics,
            )

            callback(context)
            pbar_train.set_description("Total Epochs (Training ...)")

        # bring back the optimizer to train mode
        if hasattr(optimizer, "train"):
            optimizer.train()


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(conf: DictConfig):

    device = conf.default_device

    seed_everything(conf.seed)

    model: InContextModel = instantiate(conf.model.obj)
    train_meta_dataset = instantiate(conf.train_meta_dataset)  # dataset of datasets

    # load model and optimizer state if resuming training
    checkpoint, resume, wandb_run_id = None, False, None
    if (
        conf.resume_training.enabled
        and conf.resume_training.checkpoint_path is not None
    ):
        checkpoint = torch.load(
            conf.resume_training.checkpoint_path,
            weights_only=False,
            map_location=device,
        )
        run_name = checkpoint["run_name"]
        if "wandb" in run_name:
            wandb_run_id = run_name.split("-")[-1]
        resume = True

    if conf.wandb.enabled:
        wandb_run_name = (
            str(conf.wandb.run_name) if conf.wandb.run_name is not None else None
        )
        tags = (
            [f"{key}:{value}" for key, value in conf.wandb.tags.items()]
            if "tags" in conf.wandb
            else []
        )
        wandb.init(
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf, resolve=True),
            name=None if resume and wandb_run_id is not None else wandb_run_name,
            tags=tags,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
            id=wandb_run_id,
            resume="must" if resume and wandb_run_id is not None else "never",
        )
    if "callbacks" not in conf:
        callbacks = []
    else:
        callbacks = _order_callbacks(
            [instantiate(callback) for callback in conf.callbacks.values()]
        )

    # set the default name for the checkpoint callback so it will be saved in the same directory
    for callback in callbacks:
        if isinstance(callback, Checkpoint) and resume:
            callback.default_name = run_name

    train(
        model=model,
        optimizer_partial=instantiate(conf.optimizer),
        train_meta_dataset=train_meta_dataset,
        callbacks=callbacks,
        lr_scheduler_partial=instantiate(conf.lr_scheduler)
        if "lr_scheduler" in conf
        else None,
        compile=conf.compile,
        num_workers=conf.num_workers,
        prefetch_factor=conf.prefetch_factor,
        checkpoint=checkpoint,
        wandb_enabled=conf.wandb.enabled,
        device=device,
        **OmegaConf.to_container(instantiate(conf.trainer)),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
