import math
from typing import List, Tuple, Union

import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def plot_km_hist(data, data_name, colors: list[str] = None):
    """
    Plot Kaplan-Meier survival curve and histogram of event/censor times.
    Args:
        data (pd.DataFrame): DataFrame containing 'time' and 'event' columns.
        data_name (str): Name of the dataset for the plot title.
        colors (list[str], optional): List of 3 colors for the plot. Defaults to None.
            First color is for KM curve, second for event histogram, third for censor histogram.
    """
    import matplotlib.pyplot as plt

    data = data.astype({"time": "float64", "event": "int32"})

    event_times = data.time.values[data.event.values == 1]
    censor_times = data.time.values[data.event.values == 0]

    # Sturges formula
    intervals = math.ceil(math.log2(data.shape[0]) + 1)
    bins = np.linspace(0, round(data.time.max()), intervals)

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    km = KaplanMeierFitter()
    km.fit(data.time.values, data.event.values)

    # plot the Kaplan-Meier and 95% confidence interval
    if colors is None:
        colors = [
            "#6C8EBF",
            "#82B366",
            "#D6B656",
        ]  # Default colors: blue, orange, green
    ax0 = km.plot_survival_function(
        ax=ax0, color=colors[0], linewidth=2.5, ci_show=False
    )
    ax0.fill_between(
        km.survival_function_.index,
        km.confidence_interval_.iloc[:, 0],
        km.confidence_interval_.iloc[:, 1],
        color=colors[0],
        alpha=0.3,
        label="95% CI",
        zorder=1,
        step="post",
    )
    ax0.set_xlabel("Time", color="black", weight="bold")
    ax0.set_ylabel("Survival Probability", color="black", weight="bold")
    ax0.set_xlim(0, round(data.time.max()))
    ax0.set_ylim(0, 1.05)
    ax0.set_title(f"{data_name}", fontsize=12, weight="bold")
    ax0.tick_params(axis="x", labelcolor="black", labelsize=10, width=1.5)
    ax0.tick_params(axis="y", labelcolor="black", labelsize=10, width=1.5)
    ax0.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    ax0.set_axisbelow(True)
    # remove legend for KM plot
    ax0.legend().remove()

    ax1 = ax0.twinx()
    ax1.hist(
        [event_times, censor_times],
        bins=bins,
        histtype="barstacked",
        stacked=True,
        alpha=1,
        color=[colors[1], colors[2]],
        zorder=2,
    )
    # ax1.set_yscale('log')
    ax1.set_ylabel("Counts", color="black", weight="bold")
    ax1.legend(["Event", "Censored"], loc="best")
    ax1.yaxis.grid(False)
    ax0.set_zorder(ax1.get_zorder() + 1)
    ax0.patch.set_visible(False)

    # ax1.set_title("Event/Censor Time Histogram")

    # fig.set_size_inches(12, 12)
    # plt.suptitle(
    #     '{}\n #Subjects: {}; %Censoring: {:.1f}%'.format(data_rename[data_name], data.shape[0], round(censor_rate * 100, 3))
    # )
    # plt.suptitle(
    #     '{}'.format(data_rename[data_name])
    # )
    # plt.show()
    plt.tight_layout()
    fig.savefig(f"figs/data/{data_name}.png", dpi=400)
    plt.close(fig)


def get_ending_prob(data):
    """
    Calculate the ending survival probability of the survival data, using KM.
    """
    data = data.astype({"time": "float64", "event": "int32"})
    km = KaplanMeierFitter()
    km.fit(data.time.values, data.event.values)
    return km.survival_function_.iloc[-1, 0]


def get_preprocessor(
    type: str = "both",
    labels: Union[Tuple[str, ...], List[str]] = (
        "time",
        "event",
    ),
) -> ColumnTransformer:
    """
    Get a preprocessor for the dataset.

    Continuous features (prefix 'num_'):
        - "impute": median imputation only
        - "scale": standard scaling only
        - "both": median imputation + standard scaling

    Non-continuous features:
        - "impute": mode imputation only
        - "scale": no transformation (passthrough)
        - "both": mode imputation only

    Parameters
    ----------
    type : {"impute", "scale", "both"}, optional
        Type of preprocessor.
        "impute" includes only imputation,
        "scale" includes only scaling,
        "both" includes both imputation and scaling.
    labels : Union[Tuple[str, ...], List[str]], optional
        List of label columns to exclude from preprocessing. Default is ("time", "event").

    Returns
    -------
    ColumnTransformer
        The preprocessor.
    """
    kind = type.lower()
    if kind not in {"impute", "scale", "both"}:
        raise ValueError(
            f"Invalid type='{type}'. Expected one of 'impute', 'scale', 'both'."
        )

    # Column selectors
    sel_con = make_column_selector(pattern="^num_")

    # Avoid fragile regex negative lookaheads: select exact label names directly.
    def sel_cat(df):
        return [
            c for c in df.columns if (not c.startswith("num_")) and (c not in labels)
        ]

    # Numeric / continuous
    num_steps = []
    if kind in {"impute", "both"}:
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    if kind in {"scale", "both"}:
        num_steps.append(("scale", StandardScaler()))

    if num_steps:
        enc_con = Pipeline(steps=num_steps)
    else:
        # Should not really happen with the current choices, but kept for completeness
        enc_con = "passthrough"

    # Categorical / non-continuous
    cat_steps = []
    if kind in {"impute", "both"}:
        cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    # For "scale" we do nothing on categorical features (passthrough).

    if cat_steps:
        enc_cat = Pipeline(steps=cat_steps)
    else:
        enc_cat = "passthrough"

    # Column transformer
    enc_df = ColumnTransformer(
        transformers=[
            ("num", enc_con, sel_con),  # numeric features
            ("cat", enc_cat, sel_cat),  # categorical features
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    enc_df.set_output(transform="pandas")
    return enc_df
