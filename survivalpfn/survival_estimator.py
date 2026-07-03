import math
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from scipy.stats import chisquare
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from .models import InContextModel
from .models.utils import HistogramDistribution

DEFAULT_MODEL_REPO_ID = "shi-ang/SurvivalPFN"
_PREFERRED_CHECKPOINT_FILES = (
    "survivalpfn_v0.1.pt",
    "survivalpfn_v0.pt",
    "latest.pt",
    "best.pt",
    "checkpoint.pt",
    "model.pt",
    "pytorch_model.bin",
)


def is_hf_model_path(model_path: str) -> bool:
    """
    Check if a given path is a Hugging Face model path (repo_id/model_name).

    Args:
        model_path: The model path to check

    Returns:
        bool: True if it's a Hugging Face model path, False otherwise
    """
    # Check if path doesn't exist locally but follows the pattern of org/repo or user/repo
    if (
        not os.path.exists(model_path)
        and "/" in model_path
        and model_path.count("/") == 1
    ):
        return True
    return False


def _select_hf_checkpoint_file(
    model_path: str, revision: str | None = None, model_filename: str | None = None
) -> str:
    if model_filename is not None:
        return model_filename
    try:
        repo_files = list_repo_files(model_path, revision=revision)
    except (RepositoryNotFoundError, HfHubHTTPError):
        # Keep the historical filename as a fallback. hf_hub_download will raise
        # the actionable Hub error if the repo or file is not accessible.
        return _PREFERRED_CHECKPOINT_FILES[0]

    for filename in _PREFERRED_CHECKPOINT_FILES:
        if filename in repo_files:
            return filename

    checkpoint_files = sorted(
        filename
        for filename in repo_files
        if filename.endswith((".pt", ".pth", ".ckpt", ".bin"))
    )
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]
    if checkpoint_files:
        raise ValueError(
            f"Multiple checkpoint files found in Hugging Face repo {model_path}: "
            f"{checkpoint_files}. Use a local path or keep a single release checkpoint in the repo."
        )
    raise ValueError(
        f"No PyTorch checkpoint file found in Hugging Face repo {model_path}."
    )


def download_from_hf_hub(
    model_path: str,
    cache_dir: str,
    model_filename: str | None = None,
    model_revision: str | None = None,
) -> str:
    """
    Download a model from the Hugging Face Hub.

    Args:
        model_path: The model path in format 'org/repo' or 'user/repo'
        cache_dir: Optional directory to cache the downloaded model

    Returns:
        str: Path to the downloaded model file
    """
    filename = _select_hf_checkpoint_file(
        model_path, revision=model_revision, model_filename=model_filename
    )
    local_path = hf_hub_download(
        repo_id=model_path,
        filename=filename,
        cache_dir=cache_dir,
        revision=model_revision,
    )
    return local_path


class SurvivalEstimator:
    """

    Args:
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
        model_path (str): The path to the model checkpoint. Can be:
            - A local file path
            - A Hugging Face model path
        model_filename (str, optional): Checkpoint filename when `model_path` is a
            Hugging Face repo id. Defaults to the release checkpoint selection order.
        model_revision (str, optional): Hugging Face revision, branch, tag, or commit.
        icl_model (InContextModel, optional): An already loaded InContextModel instance.
            Only will be used if `model_path = None`.
        cache_dir (str, optional): Directory to cache downloaded models from Hugging Face.
            Defaults to ~/.cache/survivalpfn.
        calibrate (bool): Whether to calibrate the model's temperature using n-fold cross-validation.
        n_folds (int): The number of folds used for temperature calibration.
        calibrate_T_min (float): The minimum temperature to use in calibration.
        calibrate_T_max (float): The maximum temperature to use in calibration.
        calibrate_T_size (int): The number of temperature values to search during calibration.
        calibrate_T_batch_size (int): The number of temperature values to evaluate per calibration batch.
        verbose (bool): Whether to print progress messages for the `predict_cep` function.
    """

    def __init__(
        self,
        device: str,
        model_path: str | None = DEFAULT_MODEL_REPO_ID,
        model_filename: str | None = None,
        model_revision: str | None = None,
        verbose: bool = False,
        cache_dir: str | None = None,
        icl_model: InContextModel | None = None,
        calibrate: bool = False,
        n_folds: int = 3,
        calibrate_T_min: float = 0.001,
        calibrate_T_max: float = 10.0,
        calibrate_T_size: int = 500,
        calibrate_T_batch_size: int = 50,
    ):
        self.model_path = model_path
        self.model_filename = model_filename
        self.model_revision = model_revision
        self.model_source_metadata: dict[str, str | None] = {}
        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else os.path.join(Path.home(), ".cache", "survivalpfn")
        )
        self.icl_model: InContextModel = icl_model

        self.device = device
        self.calibrate = calibrate
        self.n_folds = n_folds
        self.calibrate_T_min = calibrate_T_min
        self.calibrate_T_max = calibrate_T_max
        self.calibrate_T_size = calibrate_T_size
        self.calibrate_T_batch_size = calibrate_T_batch_size

        # The maximum number of features to use for the model. If the number of features are
        # larger than this value, the model will apply PCA to reduce the dimensionality.
        self.max_feature_size = None
        self.x_dim_imputer = None
        self.x_dim_transformer = (
            FunctionTransformer()
        )  # identity transformer by default

        self.X_train, self.delta_train, self.T_train = None, None, None
        self._raw_feature_dim = None
        self.temperature = 1.0
        self.prediction_temperature = 1.0

        self.verbose = verbose

    def _check_fitted(self):
        if (
            self.X_train is None
            or self.delta_train is None
            or self.T_train is None
            or self.icl_model is None
        ):
            raise ValueError(
                "The estimator must be fitted before calling the estimate function."
            )

    def _transform_query_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform query features to match the representation used during fitting.
        """
        if self._raw_feature_dim is None or self.X_train is None:
            raise ValueError(
                "The estimator must be fitted before transforming query features."
            )

        fitted_dim = self.X_train.shape[1]
        if self._raw_feature_dim > fitted_dim:
            if X.shape[1] == self._raw_feature_dim:
                X_for_transform = X
                if self.x_dim_imputer is not None:
                    X_for_transform = self.x_dim_imputer.transform(X_for_transform)
                return self.x_dim_transformer.transform(X_for_transform)
            if X.shape[1] == fitted_dim:
                return X
            raise ValueError(
                f"Query feature dimension must be either {self._raw_feature_dim} (raw) or {fitted_dim} (transformed), "
                f"got {X.shape[1]}."
            )
        return X

    def load_model(self):
        """
        Load the model from the specified path or download it from Hugging Face.
        """
        if self.model_path is not None:
            model_path = self.model_path

            # Check if the model path is a Hugging Face model path
            if is_hf_model_path(model_path):
                filename = _select_hf_checkpoint_file(
                    model_path,
                    revision=self.model_revision,
                    model_filename=self.model_filename,
                )
                model_path = download_from_hf_hub(
                    model_path,
                    self.cache_dir,
                    model_filename=filename,
                    model_revision=self.model_revision,
                )
                self.model_source_metadata = {
                    "repo_id": self.model_path,
                    "filename": filename,
                    "revision": self.model_revision,
                    "local_path": model_path,
                }
            else:
                self.model_source_metadata = {
                    "repo_id": None,
                    "filename": None,
                    "revision": None,
                    "local_path": model_path,
                }

            # Load the model from the local path
            ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
            model_state = ckpt.get("model_state_dict", ckpt.get("model"))
            config = ckpt.get("model_config", ckpt.get("cfg"))
            if model_state is None or config is None:
                raise KeyError(
                    "Checkpoint must contain either ('model_state_dict', "
                    "'model_config') or ('model', 'cfg')."
                )

            self.icl_model = InContextModel.load(
                model_state=model_state, model_config=config
            ).to(self.device)
        elif self.icl_model is not None:
            # If icl_model is provided, use it directly
            self.icl_model.to(self.device)
            config = self.icl_model.model_config
        else:
            raise ValueError("Either model_path or icl_model must be provided.")

        if "model_type" not in config or config["model_type"] == "tabdpt":
            self.max_feature_size = config["model"]["max_num_features"] - 1
            self.x_dim_transformer = TruncatedSVD(
                n_components=self.max_feature_size, algorithm="arpack"
            )

    @torch.no_grad()
    def _predict_distribution(
        self,
        X_context: np.ndarray,
        delta_context: np.ndarray,
        T_context: np.ndarray,
        X_query: np.ndarray,
        temperature: float | torch.Tensor,
    ) -> HistogramDistribution:
        if self.icl_model is None:
            raise ValueError(
                "SurvivalEstimator must be fitted before calling _predict_distribution."
            )

        squeeze_temperature_dim = not isinstance(temperature, torch.Tensor)
        if isinstance(temperature, torch.Tensor):
            temperatures = temperature.to(device=self.device, dtype=torch.float32)
            if temperatures.dim() == 0:
                temperatures = temperatures.reshape(1)
                squeeze_temperature_dim = True
            elif temperatures.dim() != 1:
                raise ValueError(
                    f"temperature must be a scalar or 1D tensor, got shape {tuple(temperatures.shape)}."
                )
        else:
            temperatures = torch.tensor(
                [temperature], device=self.device, dtype=torch.float32
            )

        self.icl_model.eval()

        logits, bin_centers, bin_edges = self.icl_model.predict(
            # shape: (1, context_size, num_features)
            X_context=torch.from_numpy(X_context)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0),
            # shape: (1, context_size)
            delta_context=torch.from_numpy(delta_context)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0),
            T_context=torch.from_numpy(T_context)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0),
            # shape: (1, query_size, num_features)
            X_query=torch.from_numpy(X_query)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0),
            temperature=temperatures,
        )  #  (1, num_temperatures, query_size, num_bins), (...)

        logits = logits.squeeze(0)
        bin_edges = bin_edges.squeeze(0)
        bin_centers = bin_centers.squeeze(0)
        if squeeze_temperature_dim:
            logits = logits.squeeze(0)
            bin_edges = bin_edges.squeeze(0)
            bin_centers = bin_centers.squeeze(0)

        return HistogramDistribution(
            logits=logits,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )

    @torch.no_grad()
    def _calculate_dcalibration_scores(
        self,
        X: np.ndarray,
        delta: np.ndarray,
        T: np.ndarray,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        temperatures = temperatures.to(device=self.device, dtype=torch.float32)
        scores = torch.zeros_like(temperatures, device=self.device)
        all_indices = np.arange(X.shape[0])
        fold_indices = np.array_split(all_indices, min(self.n_folds, X.shape[0]))

        for val_idx in fold_indices:
            if len(val_idx) == 0:
                continue
            train_idx = np.setdiff1d(all_indices, val_idx)
            if len(train_idx) == 0:
                continue

            dist = self._predict_distribution(
                X_context=X[train_idx],
                delta_context=delta[train_idx],
                T_context=T[train_idx],
                X_query=X[val_idx],
                temperature=temperatures,
            )

            obs_time = torch.from_numpy(T[val_idx]).to(
                device=self.device, dtype=torch.float32
            )
            obs_time = obs_time.unsqueeze(0).expand(temperatures.shape[0], -1)
            survival_probs = dist.survival_at(obs_time).detach().cpu().numpy()

            from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration

            fold_scores = []
            for i in range(temperatures.shape[0]):
                _, _, dcal_hist = d_calibration(
                    survival_probs[i], delta[val_idx], num_bins=10
                )
                dcal_stats, _ = chisquare(dcal_hist)
                fold_scores.append(float(dcal_stats))
            scores += torch.tensor(fold_scores, device=self.device, dtype=torch.float32)

        return scores / len(fold_indices)

    def _calibrate_fn(self, X: np.ndarray, delta: np.ndarray, T: np.ndarray) -> None:
        """
        Calibrate temperature with n-fold cross-validation using the D-calibration statistic.
        """
        temperatures = torch.logspace(
            start=np.log10(self.calibrate_T_min),
            end=np.log10(self.calibrate_T_max),
            steps=self.calibrate_T_size,
            device=self.device,
            dtype=torch.float32,
        )

        all_scores = []
        num_batches = math.ceil(len(temperatures) / self.calibrate_T_batch_size)
        pbar = tqdm(
            range(num_batches), desc="Calibrating temperature", disable=not self.verbose
        )
        for i in pbar:
            start_idx = i * self.calibrate_T_batch_size
            end_idx = min((i + 1) * self.calibrate_T_batch_size, len(temperatures))
            batch_temperatures = temperatures[start_idx:end_idx]
            batch_scores = self._calculate_dcalibration_scores(
                X=X,
                delta=delta,
                T=T,
                temperatures=batch_temperatures,
            )
            all_scores.append(batch_scores)

        scores = torch.cat(all_scores, dim=0)
        best_idx = torch.argmin(scores).item()
        self.temperature = temperatures[best_idx].item()

    def fit(
        self, X: np.ndarray, delta: np.ndarray, T: np.ndarray
    ) -> "SurvivalEstimator":
        """
        Fit the model using the provided data.

        Args:
            X (np.ndarray): The covariate data with shape [N, D].
            delta (np.ndarray): The event indicator data with shape [N], where 1 indicates the event occurred and 0 indicates censoring.
            T (np.ndarray): The time of event or censoring data with shape [N].
        """
        self.temperature = 1.0

        # load the model
        self.load_model()

        self._raw_feature_dim = X.shape[1]

        # set the x_dim_transform and transform the data
        if self.max_feature_size is not None and X.shape[1] > self.max_feature_size:
            self.x_dim_imputer = SimpleImputer(strategy="median")
            X = self.x_dim_imputer.fit_transform(X)
            X = self.x_dim_transformer.fit_transform(X)

        self.X_train = X
        self.delta_train = delta
        self.T_train = T

        if self.calibrate:
            self._calibrate_fn(X=self.X_train, delta=self.delta_train, T=self.T_train)

        return self

    def predict_event_distribution(self, X: np.ndarray) -> HistogramDistribution:
        """
        Predict event-time distribution for each query sample.

        Args:
            X (np.ndarray): Covariate data with shape [N, D].

        Returns:
            HistogramDistribution: Predicted histogram distribution for each query sample.
        """
        self._check_fitted()
        X_query = self._transform_query_features(X)
        return self._predict_distribution(
            X_context=self.X_train,
            delta_context=self.delta_train,
            T_context=self.T_train,
            X_query=X_query,
            temperature=self.temperature,
        )

    def predict_event_time(self, X: np.ndarray, type: str = "median") -> np.ndarray:
        """
        Predicts the survival time associated with the given covariates.
        Args:
            X (np.ndarray): The covariate data with shape [N, D].
            type (str): The type of time to predict. Can be "median", "mode", or "rmst".

        Returns:
            np.ndarray: The predicted survival times with shape [N]. (E[E | X])
        """
        dist = self.predict_event_distribution(X)
        if type == "median":
            return dist.median().cpu().numpy()
        elif type == "mode":
            return dist.mode().cpu().numpy()
        elif type == "rmst":
            return dist.rmst().cpu().numpy()
        else:
            raise ValueError(f"Unknown time prediction type: {type}")

    def event_density_at_obs(
        self, X: np.ndarray, obs_event_time: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the density function for the given covariates and time points.
        Args:
            X (np.ndarray): The covariate data with shape [N, D].
            obs_event_time (np.ndarray): The observed event times with shape [N].
        Returns:
            np.ndarray: The predicted density function values with shape [N]. (f[E | X])
        """

        self._check_fitted()
        if obs_event_time.shape[0] != X.shape[0]:
            raise ValueError(
                f"obs_event_time must have the same number of samples as X, got {obs_event_time.shape[0]} and {X.shape[0]}"
            )

        dist = self.predict_event_distribution(X)

        obs_event_time = torch.from_numpy(obs_event_time).to(
            device=self.device, dtype=torch.float32
        )
        density_values = dist.density_at(obs_event_time)
        return density_values.cpu().numpy()

    def survival_at_observed_time(
        self, X: np.ndarray, obs_event_time: np.ndarray
    ) -> np.ndarray:
        """
        Predict the survival probability evaluated at observed times for each sample.

        Args:
            X (np.ndarray): Covariate data with shape [N, D].
            obs_event_time (np.ndarray): Observed event or censoring times with shape [N].

        Returns:
            np.ndarray: Predicted survival probabilities at observed times with shape [N].
        """
        self._check_fitted()
        if obs_event_time.shape[0] != X.shape[0]:
            raise ValueError(
                f"obs_event_time must have the same number of samples as X, got {obs_event_time.shape[0]} and {X.shape[0]}"
            )

        dist = self.predict_event_distribution(X)
        obs_event_time = torch.from_numpy(obs_event_time).to(
            device=self.device, dtype=torch.float32
        )
        return dist.survival_at(obs_event_time).cpu().numpy()

    def S(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predicts the survival function for the given covariates and time points.
        Args:
            X (np.ndarray): The covariate data with shape [N, D].
            t (np.ndarray): The time points at which to evaluate the survival function with shape [T].
        Returns:
            np.ndarray: The predicted survival function values with shape [N, T]. (Pr[E > t | X])
        """
        dist = self.predict_event_distribution(X)
        t = torch.from_numpy(t).to(device=self.device, dtype=torch.float32)
        survival_probs = dist.survival_function(t)
        return survival_probs.cpu().numpy()

    def h(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predicts the hazard function for the given covariates and time points.
        Args:
            X (np.ndarray): The covariate data with shape [N, D].
            t (np.ndarray): The time points at which to evaluate the hazard function with shape [T].
        Returns:
            np.ndarray: The predicted hazard function values with shape [N, T].
        """
        dist = self.predict_event_distribution(X)
        t = torch.from_numpy(t).to(device=self.device, dtype=torch.float32)
        hazard_values = dist.hazard_function(t)
        return hazard_values.cpu().numpy()
