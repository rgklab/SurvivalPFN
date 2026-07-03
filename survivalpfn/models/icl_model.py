from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn

from .constants import EPSILON_STABILITY
from .loading import load_pretrained_tabdpt_model
from .model import TabDPTLongContextModel
from .utils import (
    QuantileTransformState,
    forward_KL_hl_gaussian_loss,
    forward_KL_one_hot_loss,
    lognormal_transform,
    pad_x,
    quantile_grids,
    quantile_transform,
    surv_nll,
)


class InContextModel(nn.Module):
    def __init__(
        self,
        model: TabDPTLongContextModel,
        model_config: dict,  # the config containing the constructor arguments for the model
        sigma: float = 0.5,
        vmin: float = -5.0,
        vmax: float = 5.0,
        query_strategy: str = "random",  # "random", "event", "both", or "both_fix_len".
        time_transform: str = "quantile",  # "quantile" or "lognormal"
    ):
        super().__init__()
        self.model: nn.Module = model
        if time_transform not in {"quantile", "lognormal"}:
            raise ValueError(
                f"Unknown time_transform {time_transform}. "
                "Supported transforms are 'quantile' and 'lognormal'."
            )

        effective_vmin = 0.0 if time_transform == "quantile" else vmin
        effective_vmax = 1.0 if time_transform == "quantile" else vmax

        model_config["sigma"] = sigma
        model_config["vmin"] = effective_vmin
        model_config["vmax"] = effective_vmax
        model_config["query_strategy"] = query_strategy
        model_config["time_transform"] = time_transform

        self.model_config = model_config

        self.prepare_input = lambda x, y: (pad_x(x, model.num_features), y)

        self.nbins = model_config["model"]["nbins"]
        self.sigma = sigma
        self.loss_sigma = 0.05 if time_transform == "quantile" else sigma

        # NOTE: These variables are stored to avoid re-initializing them for each forward pass
        self.vmin = effective_vmin
        self.vmax = effective_vmax
        self.time_transform = time_transform

        # Here we split nbins-1 bins within the [vmin, vmax) range,
        # the 1 extra bin is used for the residual mass in [vmax, inf), which is important for non-trivial tail mass.
        bin_edges = torch.linspace(self.vmin, self.vmax, self.nbins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + 0.5 * bin_width  # shape: (nbins - 1,)

        self.register_buffer("bin_edges", bin_edges)  # (nbins,)
        self.register_buffer("bin_width", bin_width)  # () – 0-D tensor
        self.register_buffer("bin_centers", bin_centers)  # (nbins - 1,)

        self.query_strategy = query_strategy
        if self.query_strategy == "random":
            if self.loss_sigma > 0:
                self.loss_fn = lambda logits, y_target: forward_KL_hl_gaussian_loss(
                    logits=logits,
                    y_target=y_target,
                    sigma=self.loss_sigma,
                    bin_edges=self.bin_edges,
                    vmin=self.vmin,
                    vmax=self.vmax,
                )
            else:
                self.loss_fn = lambda logits, y_target: forward_KL_one_hot_loss(
                    logits=logits,
                    y_target=y_target,
                    bin_centers=self.bin_centers,
                    vmin=self.vmin,
                    vmax=self.vmax,
                )
        elif self.query_strategy in ["event", "both", "both_fix_len"]:
            self.loss_fn = lambda logits, t_target, delta_target: surv_nll(
                logits=logits,
                t_target=t_target,
                delta_target=delta_target,
                bin_edges=self.bin_edges,
            )
        else:
            raise ValueError(
                f"Unknown query strategy {self.query_strategy}. Supported strategies are 'random', 'event', 'both', and 'both_fix_len'."
            )

    def _get_shift_scale(self, time_context: torch.Tensor):
        return torch.mean(time_context, dim=1), torch.std(
            time_context, dim=1
        ) + EPSILON_STABILITY

    def _transform_time_to_model_space(
        self,
        samples: torch.Tensor,
        shift: torch.Tensor | None,
        scale: torch.Tensor | None,
        quantile_state: QuantileTransformState | None,
    ) -> torch.Tensor:
        if self.time_transform == "lognormal":
            return lognormal_transform(
                samples=samples,
                mean=shift,
                std=scale,
                type="lognormal2normal",
            )
        elif self.time_transform == "quantile":
            return quantile_transform(
                samples=samples,
                state=quantile_state,
                type="time2quantile",
            )
        else:
            raise ValueError(
                f"Unknown time_transform {self.time_transform}. "
                "Supported transforms are 'quantile' and 'lognormal'."
            )

    def _transform_time_from_model_space(
        self,
        samples: torch.Tensor,
        shift: torch.Tensor | None,
        scale: torch.Tensor | None,
        quantile_state: QuantileTransformState | None,
    ) -> torch.Tensor:
        if self.time_transform == "lognormal":
            return lognormal_transform(
                samples=samples,
                mean=shift,
                std=scale,
                type="normal2lognormal",
            )
        elif self.time_transform == "quantile":
            return quantile_transform(
                samples=samples,
                state=quantile_state,
                type="quantile2time",
            )
        else:
            raise ValueError(
                f"Unknown time_transform {self.time_transform}. "
                "Supported transforms are 'quantile' and 'lognormal'."
            )

    def survival_losses(
        self,
        X_context: torch.Tensor,
        delta_context: torch.Tensor,  # delta = 1 if event, 0 if censored
        T_context: torch.Tensor,
        X_query: torch.Tensor,
        E_query: torch.Tensor,
        C_query: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:

        # Avoid broken autograd when inplace operations are performed
        T_context = T_context + EPSILON_STABILITY  # avoid zero times
        E_query = E_query + EPSILON_STABILITY  # avoid zero times
        C_query = C_query + EPSILON_STABILITY  # avoid zero times

        if self.time_transform == "lognormal":
            shift, scale = self._get_shift_scale(T_context)
            quantile_state = None
        elif self.time_transform == "quantile":
            shift, scale = None, None
            # The quantile grid depends only on T_context, so build it once and
            # reuse it for all target transformations in this loss call.
            quantile_state = quantile_grids(time_context=T_context)
        else:
            raise ValueError(
                f"Unknown time_transform {self.time_transform}. "
                "Supported transforms are 'quantile' and 'lognormal'."
            )

        x_and_delta_context = torch.cat(
            [
                delta_context.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)
        T_standardized = self._transform_time_to_model_space(
            T_context, shift, scale, quantile_state
        )

        if self.query_strategy == "random":
            # We randomly ask the model to predict either the event or censor distribution (delta_query)
            # The outcome is the true event or censor time corresponding to the chosen delta_query.
            # Therefore, we can use the loss as the outcome is always observed
            event_prob = (
                (E_query < C_query).to(dtype=X_query.dtype).mean(dim=1, keepdim=True)
            )
            delta_query = torch.bernoulli(event_prob.expand_as(E_query)).to(
                dtype=X_query.dtype
            )  # shape: (batch_size, query_len)
            x_and_delta_query = torch.cat(
                [
                    delta_query.unsqueeze(-1),
                    X_query,
                ],
                dim=2,
            )  # shape: (batch_size,  query_len , num_features + 1)
            T_query = torch.where(
                delta_query.bool(), E_query, C_query
            )  # shape: (batch_size, query_len), use E_query if event, C_query if censored
        elif self.query_strategy == "event":
            # We always ask the model to predict event distribution (delta in x_and_delta_query is set to 1).
            # The outcome is the right-censored time according to the true event/censor time, and event indicator recorded by delta_query.
            # Therefore, survival NLL is used as the loss.
            delta_query = (E_query < C_query).to(
                dtype=E_query.dtype
            )  # shape: (batch_size, query_len)
            x_and_delta_query = torch.cat(
                [
                    torch.ones_like(delta_query).unsqueeze(-1),
                    X_query,
                ],
                dim=2,
            )  # shape: (batch_size,  query_len , num_features + 1)
            T_query = torch.minimum(E_query, C_query)  # shape: (batch_size, query_len)
        elif self.query_strategy == "both":
            # We ask the model to predict both event and censor distribution (delta_query is set to 1 for event prediction and 0 for censor prediction).
            # The outcome is the true event or censor time corresponding to the chosen delta_query.
            # Therefore, survival NLL is used as the loss.
            delta_query_event = (E_query < C_query).to(
                dtype=E_query.dtype
            )  # shape: (batch_size, query_len)
            delta_query_censor = 1 - delta_query_event
            delta_query = torch.cat(
                [delta_query_event, delta_query_censor], dim=1
            )  # shape: (batch_size, 2 * query_len)
            query_input_delta = torch.cat(
                [
                    torch.ones_like(delta_query_event),
                    torch.zeros_like(delta_query_event),
                ],
                dim=1,
            )  # shape: (batch_size, 2 * query_len)
            X_query_twice = torch.cat(
                [X_query, X_query], dim=1
            )  # shape: (batch_size, 2 * query_len, num_features)
            x_and_delta_query = torch.cat(
                [query_input_delta.unsqueeze(-1), X_query_twice], dim=2
            )  # shape: (batch_size, 2 * query_len , num_features + 1)
            T_query = torch.minimum(E_query, C_query)  # shape: (batch_size, query_len)
            T_query = torch.cat(
                [T_query, T_query], dim=1
            )  # shape: (batch_size, 2 * query_len)
        elif self.query_strategy == "both_fix_len":
            # Like "both", we train event and censor distribution queries with survival NLL,
            # but we choose exactly one process query per original query token so the total
            # sequence length stays at context_len + query_len.
            delta_query_event = (E_query < C_query).to(
                dtype=E_query.dtype
            )  # shape: (batch_size, query_len)
            delta_query_censor = 1 - delta_query_event
            event_prob = delta_query_event.to(dtype=X_query.dtype).mean(
                dim=1, keepdim=True
            )
            query_input_delta = torch.bernoulli(event_prob.expand_as(E_query)).to(
                dtype=X_query.dtype
            )  # shape: (batch_size, query_len)
            x_and_delta_query = torch.cat(
                [query_input_delta.unsqueeze(-1), X_query], dim=2
            )  # shape: (batch_size, query_len, num_features + 1)
            T_query = torch.minimum(E_query, C_query)  # shape: (batch_size, query_len)
            delta_query = torch.where(
                query_input_delta.bool(), delta_query_event, delta_query_censor
            )
        else:
            raise ValueError(
                f"Unknown query strategy {self.query_strategy}. Supported strategies are 'random', 'event', 'both', and 'both_fix_len'."
            )

        T_query_standardized = self._transform_time_to_model_space(
            T_query, shift, scale, quantile_state
        )

        x_and_delta = torch.cat(
            [x_and_delta_context, x_and_delta_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)

        # x_src is training+test inputs with
        # shape: (batch_size, context_len + query_len, num_features + 1 (+ dummy to fit the maximum number of features))
        # t_src is training labels with shape: (batch_size, context_len)
        x_src, t_src = self.prepare_input(x_and_delta, T_standardized)

        # logits are the test labels
        logits = self.model(x_src.transpose(0, 1), t_src.transpose(0, 1)).transpose(
            0, 1
        )  # shape: (batch_size, query_len, nbins)
        logits = logits[
            :, :, -self.model.nbins :
        ]  # only keep the last nbins, which are the predictions

        logits /= temperature  # Apply temperature scaling

        if self.query_strategy == "random":
            out = self.loss_fn(
                logits=logits, y_target=T_query_standardized
            )  # shape: (batch_size, query_len)
        elif self.query_strategy in ["event", "both", "both_fix_len"]:
            out = self.loss_fn(
                logits=logits, t_target=T_query_standardized, delta_target=delta_query
            )  # shape: (batch_size, query_len)
        return out.mean(dim=-1)

    def forward(
        self,
        X_context: torch.Tensor,
        delta_context: torch.Tensor,
        T_context: torch.Tensor,
        X_query: torch.Tensor,
        E_query: torch.Tensor,
        C_query: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        The forward method simply calls survival_losses and returns the training loss.
        """
        return self.survival_losses(
            X_context=X_context,
            delta_context=delta_context,
            T_context=T_context,
            X_query=X_query,
            E_query=E_query,
            C_query=C_query,
            temperature=temperature,
        )

    def predict(
        self,
        X_context: torch.Tensor,
        delta_context: torch.Tensor,
        T_context: torch.Tensor,
        X_query: torch.Tensor,
        temperature: torch.Tensor,  # shape: (num_temperatures, )
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        T_context = T_context + EPSILON_STABILITY  # avoid zero times

        if self.time_transform == "lognormal":
            shift, scale = self._get_shift_scale(
                T_context
            )  # shape: (batch_size, ), (batch_size, )
            quantile_state = None
        elif self.time_transform == "quantile":
            shift, scale = None, None
            # Build the empirical grid once; it is reused below for context labels,
            # bin centers, and bin edges.
            quantile_state = quantile_grids(time_context=T_context)
        else:
            raise ValueError(
                f"Unknown time_transform {self.time_transform}. "
                "Supported transforms are 'quantile' and 'lognormal'."
            )

        T_standardized = self._transform_time_to_model_space(
            T_context, shift, scale, quantile_state
        )

        event_query_delta = torch.ones(X_query.shape[:-1], device=X_query.device)

        x_and_delta_context = torch.cat(
            [
                delta_context.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)

        x_and_delta_query = torch.cat(
            [
                event_query_delta.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)

        x_and_delta = torch.cat(
            [x_and_delta_context, x_and_delta_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)

        # x_src is training+test inputs with
        # shape: (batch_size, context_len + query_len, num_features + 1 (+ dummy to fit the maximum number of features))
        # t_src is training labels with shape: (batch_size, context_len)
        x_src, t_src = self.prepare_input(x_and_delta, T_standardized)

        logits = self.model(x_src.transpose(0, 1), t_src.transpose(0, 1)).transpose(
            0, 1
        )  # shape: (batch_size, query_len, nbins)
        logits = logits[
            :, :, -self.model.nbins :
        ]  # only keep the last nbins, which are the predictions

        logits = logits.unsqueeze(1)  # shape: (batch_size, 1, query_len, nbins)

        temperature = temperature[
            None, :, None, None
        ]  # shape: (1, num_temperatures, 1, 1)

        # Apply temperature scaling with shape: (batch_size, num_temperatures, query_len, nbins)
        logits = logits / temperature

        batch_size, num_temperatures, query_len = logits.shape[:3]
        bin_centers = self.bin_centers.view(1, 1, 1, -1).expand(
            batch_size, num_temperatures, query_len, -1
        )  # shape: (batch_size, num_temperatures, query_len, nbins - 1)
        bin_edges = self.bin_edges.view(1, 1, 1, -1).expand(
            batch_size, num_temperatures, query_len, -1
        )  # shape: (batch_size, num_temperatures, query_len, nbins)
        if self.time_transform == "lognormal":
            shift = shift[:, None, None].expand(batch_size, num_temperatures, query_len)
            scale = scale[:, None, None].expand(batch_size, num_temperatures, query_len)

        bin_centers = self._transform_time_from_model_space(
            bin_centers, shift, scale, quantile_state
        )
        bin_edges = self._transform_time_from_model_space(
            bin_edges, shift, scale, quantile_state
        )

        return logits, bin_centers, bin_edges

    def get_param_groups(self):
        """
        Return optimizer-specific parameter groups based on the type of the model used.
        This will sometimes help with stabilizing the training.
        """

        if isinstance(self.model, TabDPTLongContextModel):
            return [
                dict(
                    params=self.model.transformer_encoder.parameters(),
                ),
                dict(
                    params=[
                        p
                        for n, p in self.model.named_parameters()
                        if not n.startswith("transformer_encoder")
                    ],
                    weight_decay=0.0,  # no weight decay for these params
                ),
            ]
        else:
            return self.model.parameters()

    @classmethod
    def load(cls, model_state: dict, model_config: dict) -> "InContextModel":
        model_config = _normalize_model_config(
            model_config=model_config, model_state=model_state
        )
        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        inner_model_state = {
            k.replace("model.", ""): v
            for k, v in model_state.items()
            if k.startswith("model.")
        }
        if not inner_model_state:
            inner_model_state = {
                k: v
                for k, v in model_state.items()
                if k not in {"bin_edges", "bin_width", "bin_centers"}
            }
        if "model_type" not in model_config:
            model_config["model_type"] = "tabdpt"
        if model_config["model_type"] == "tabdpt":
            ckpt_loaded = {"cfg": model_config, "model": inner_model_state}
            base_model = load_pretrained_tabdpt_model(ckpt_path=None, ckpt=ckpt_loaded)
        else:
            raise ValueError("Unknown model. Supported model is 'tabdpt'.")

        sigma = model_config.get("sigma", 0.5)
        vmin = model_config.get("vmin", -5.0)
        vmax = model_config.get("vmax", 5.0)
        query_strategy = model_config.get("query_strategy", "both")
        time_transform = model_config.get("time_transform", "lognormal")

        return InContextModel(
            model=base_model,
            model_config=model_config,
            sigma=sigma,
            vmin=vmin,
            vmax=vmax,
            query_strategy=query_strategy,
            time_transform=time_transform,
        )


def _normalize_model_config(model_config: dict, model_state: dict | None = None) -> dict:
    model_config = deepcopy(model_config)
    bin_edges = None
    if model_state is not None:
        bin_edges = model_state.get("bin_edges")
        if bin_edges is None:
            bin_edges = model_state.get("_orig_mod.bin_edges")
    if bin_edges is not None:
        if "vmin" not in model_config:
            model_config["vmin"] = float(bin_edges[0].detach().cpu().item())
        if "vmax" not in model_config:
            model_config["vmax"] = float(bin_edges[-1].detach().cpu().item())
    model_config.setdefault("sigma", 0.5)
    model_config.setdefault("query_strategy", "both")

    transform_aliases = {
        "lognormal2normal": "lognormal",
        "normal2lognormal": "lognormal",
        "time2quantile": "quantile",
        "quantile2time": "quantile",
    }
    legacy_transform = (
        model_config.get("time_transform")
        or model_config.get("transform")
        or model_config.get("time_transformation")
        or model_config.get("time_transform_type")
    )
    if legacy_transform is None:
        legacy_transform = "lognormal"
    model_config["time_transform"] = transform_aliases.get(
        str(legacy_transform), str(legacy_transform)
    )
    return model_config
