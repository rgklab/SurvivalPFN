import os
import random
import warnings

import numpy as np
import torch


def pad_x(X: torch.Tensor, num_features=100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros(
        (*X.shape[:-1], num_features - n_features), device=X.device
    )
    return torch.cat([X, zero_feature_padding], dim=-1)


def numpy_nanstd(x: np.ndarray, axis: int = 0):
    # Count non-NaN values
    count = np.sum(~np.isnan(x), axis=axis)

    # Replace NaNs with 0 and compute sum
    x_sum = np.nansum(x, axis=axis)

    # Compute mean
    mean = x_sum / count

    # Reshape mean to broadcast properly
    mean_broadcast = np.expand_dims(mean, axis=axis)

    # Compute squared differences, ignoring NaNs
    sq_diff = np.where(np.isnan(x), 0, (x - mean_broadcast) ** 2)

    # Compute nanstd
    return np.sqrt(np.sum(sq_diff, axis=axis) / (count - 1))


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sample_data_split(train_data_split_buckets: tuple[float, ...]) -> float:
    """Randomly sample a train/query split from the given buckets."""
    split_idx = random.randrange(len(train_data_split_buckets))
    return train_data_split_buckets[split_idx]


def prescan_data_split_buckets(
    min_train_data_split: float,
    max_train_data_split: float,
    num_buckets: int = 4,
) -> tuple[float, ...]:
    """Generate train/query split buckets for prescanning. A fixed set of splits is good for faster training."""
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be at least 1, got {num_buckets}")
    if min_train_data_split > max_train_data_split:
        raise ValueError(
            "min_train_data_split must be less than or equal to max_train_data_split, "
            f"got {min_train_data_split} > {max_train_data_split}"
        )
    if num_buckets == 1 or min_train_data_split == max_train_data_split:
        return (min_train_data_split,)

    step = (max_train_data_split - min_train_data_split) / (num_buckets - 1)
    return tuple(min_train_data_split + step * i for i in range(num_buckets))


def validate_split_strategy(
    query_strategy: str,
    min_train_data_split: float,
    max_train_data_split: float,
    n_buckets: int,
    warn: bool = True,
) -> str:
    has_fixed_train_data_split = (n_buckets == 1) or (
        min_train_data_split == max_train_data_split
    )
    if query_strategy == "both" and not has_fixed_train_data_split:
        if warn:
            warnings.warn(
                "query_strategy='both' requires a fixed train/query split because it creates variable-length ICL "
                "inputs. Falling back to query_strategy='both_fix_len' for multiple train/query split buckets.",
                UserWarning,
                stacklevel=2,
            )
        return "both_fix_len"
    return query_strategy
