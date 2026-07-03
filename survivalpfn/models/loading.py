import torch
from omegaconf import OmegaConf

from .model import TabDPTLongContextModel


class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)


def _to_plain_dict(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {
            key: _to_plain_dict(item) if isinstance(item, dict) else item
            for key, item in value.items()
        }
    return value


def _get_nested(config: dict, *keys: str, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _extract_tabdpt_state(model_state: dict) -> dict:
    model_state = {
        key.replace("_orig_mod.", ""): value for key, value in model_state.items()
    }
    if any(key.startswith("model.") for key in model_state):
        model_state = {
            key.replace("model.", "", 1): value
            for key, value in model_state.items()
            if key.startswith("model.")
        }
    return {
        key: value
        for key, value in model_state.items()
        if key not in {"bin_edges", "bin_width", "bin_centers"}
    }


def _build_tabdpt_from_config(
    model_state: dict, config: dict
) -> TabDPTLongContextModel:
    model_config = config["model"]
    dropout = float(_get_nested(config, "training", "dropout", default=0.0))
    model = TabDPTLongContextModel(
        dropout=dropout,
        n_out=model_config["max_num_classes"],
        nhead=model_config["nhead"],
        nhid=model_config["emsize"] * model_config["nhid_factor"],
        ninp=model_config["emsize"],
        nlayers=model_config["nlayers"],
        num_features=model_config["max_num_features"],
        nbins=model_config["nbins"],
    )
    model.load_state_dict(_extract_tabdpt_state(model_state))
    model.eval()
    return model


def load_pretrained_tabdpt_model(
    ckpt_path: str | None, ckpt: dict | None = None
) -> TabDPTLongContextModel:
    assert ckpt_path is not None or ckpt is not None, (
        "Either ckpt_path or ckpt must be provided."
    )
    checkpoint = (
        torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ckpt is None
        else ckpt
    )
    config = _to_plain_dict(checkpoint.get("cfg", checkpoint.get("model_config")))
    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if config is None or model_state is None:
        raise KeyError(
            "Checkpoint must contain either ('cfg', 'model') or "
            "('model_config', 'model_state_dict')."
        )
    return _build_tabdpt_from_config(model_state=model_state, config=config)


def load_pretrained_tabdpt_config(
    ckpt_path: str | None, ckpt: dict | None = None
) -> dict:
    assert ckpt_path is not None or ckpt is not None, (
        "Either ckpt_path or ckpt must be provided."
    )
    checkpoint = (
        torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ckpt is None
        else ckpt
    )
    cfg = checkpoint.get("cfg", checkpoint.get("model_config"))
    if cfg is None:
        raise KeyError("Checkpoint must contain either 'cfg' or 'model_config'.")
    return _to_plain_dict(cfg)
