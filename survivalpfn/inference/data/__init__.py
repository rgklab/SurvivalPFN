import os
import random
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy
from SurvSet.data import SurvLoader

from .util import get_ending_prob, get_preprocessor

_SURVSET_DATA = [
    "hdfail",
    "stagec",
    "veteran",
    "LeukSurv",
    "zinc",
    "burn",
    "UnempDur",
    "rott2",
    "FRTCS",
    "cost",
    "rdata",
    "Aids2",
    "divorce",
    "micro.censure",
    "cgd",
    "pharmacoSmoking",
    "uis",
    "Dialysis",
    "nki70",
    "Melanoma",
    "cancer",
    "TRACE",
    "ova",
    "smarto",
    "vlbw",
    "phpl04K8a",
    "dataDIVAT3",
    "dataDIVAT2",
    "prostateSurvival",
    "dataDIVAT1",
    "e1684",
    "acath",
    "Framingham",
    "colon",
    "mgus",
    "d.oropha.rec",
    "actg",
    "dataOvarian1",
    "hepatoCellular",
    "ovarian",
    "retinopathy",
    "Pbc3",
    "whas500",
    "diabetes",
    "glioma",
    "Unemployment",
    "oldmort",
    "Bergamaschi",
    "prostate",
    "grace",
    "scania",
]


# current file path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
METABRIC_PAPER_URL = (
    "https://raw.githubusercontent.com/shi-ang/CensoredMAE/"
    "a658a58e91f177a0593caa563d68f23d5d98b68a/"
    "data/Metabric/Metabric.csv"
)


class SurvivalBenchmarkDataset:
    def __init__(
        self,
        data_name: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        preprocess: bool = True,
        seed: int = 42,
        fixed_split: bool = False,
    ) -> None:
        self.data_name = data_name

        assert train_ratio > 0 and val_ratio >= 0 and test_ratio > 0, (
            "Check train validation test fraction."
        )
        ratio_sum = train_ratio + val_ratio + test_ratio
        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.initial_seed = seed
        self.seed = seed
        self.fixed_split = fixed_split

        # load the data and do some basic preprocessing and statistics
        self.df = load_data(self.data_name)
        assert "time" in self.df.columns and "event" in self.df.columns, (
            "The event time variable and censor indicator variable is missing or need to be renamed."
        )

        self.features = self.df.drop(columns=["time", "event"]).columns.tolist()
        self.n_features = len(self.features)
        # numerical_features are those that start with "num_", categorical_features are those that do not start with "num_"
        self.num_features = [f for f in self.features if f.startswith("num_")]
        self.cat_features = [f for f in self.features if not f.startswith("num_")]
        assert len(self.num_features) + len(self.cat_features) == self.n_features, (
            "Feature categorization error."
        )

        # some statistics
        self.n_samples = len(self.df)
        self.min_time = self.df["time"].min()
        self.max_time = self.df["time"].max()
        self.censoring_rate = 1 - self.df["event"].mean()
        self.features_dims = {
            "numerical": len(self.num_features),
            "categorical": len(self.cat_features),
            "total": len(self.features),
        }
        self.sample_feature_ratio = self.n_samples / len(self.features)
        self.missing_rate = (
            self.df.drop(columns=["time", "event"]).isnull().mean().mean()
        )
        self.ending_prob = get_ending_prob(self.df)

        if preprocess:
            self.preprocessor = get_preprocessor(type="both", labels=("time", "event"))
        else:
            self.preprocessor = None

    def __call__(
        self,
    ) -> dict:
        """Split the data into training and testing sets."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Shuffle the DataFrame
        df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        # Split the DataFrame into train and test sets
        df = df.astype(float)

        split_index = (
            int(len(df) * self.train_ratio),
            int(len(df) * (self.train_ratio + self.val_ratio)),
        )
        train_df = df.iloc[: split_index[0]]
        val_df = df.iloc[split_index[0] : split_index[1]]
        train_val_df = df.iloc[: split_index[1]]
        test_df = df.iloc[split_index[1] :]

        if self.preprocessor is not None:
            train_df = self.preprocessor.fit_transform(train_df)
            val_df = self.preprocessor.transform(val_df) if len(val_df) > 0 else val_df
            train_val_df = self.preprocessor.transform(train_val_df)
            test_df = self.preprocessor.transform(test_df)

        if not self.fixed_split:
            # update the seed for the next call to get a different split
            self.seed += 1

        return {
            "df_train": train_df,
            "df_val": val_df,
            "df_train_val": train_val_df,
            "df_test": test_df,
            "T_train": train_df["time"].values,
            "delta_train": train_df["event"].values,
            "X_train": train_df.drop(columns=["time", "event"]).values,
            "T_val": val_df["time"].values if len(val_df) > 0 else None,
            "delta_val": val_df["event"].values if len(val_df) > 0 else None,
            "X_val": val_df.drop(columns=["time", "event"]).values
            if len(val_df) > 0
            else None,
            "T_train_val": train_val_df["time"].values,
            "delta_train_val": train_val_df["event"].values,
            "X_train_val": train_val_df.drop(columns=["time", "event"]).values,
            "T_test": test_df["time"].values,
            "delta_test": test_df["event"].values,
            "X_test": test_df.drop(columns=["time", "event"]).values,
        }


ALL_DATA = sorted(
    [
        "SUPPORT",
        "METABRIC",
        "NACD",
        "FLCHAIN",
        "GBSG2",
        "NWTCO",
        "WHAS",
        "PBC",
        "GBM",
        "NPC",
        "AIDS",
        "HFCR",
        "WPBC",
        "BMT",
        "churn",
        "credit",
        "employee",
        "PDM",
        "BCCardiotox",
        "kidney_transplant",
        "larynx",
        "leukemia",
        "NCCTG",
        "lupus",
        "Rossi",
        "MSKCC",
        "COVID",
        "MIMIC-IV_all",
        "SEER_brain",
        "SEER_liver",
    ]
    + _SURVSET_DATA
)


def load_data(
    data_name: str,
) -> pd.DataFrame:
    if data_name == "SUPPORT":
        return make_support()
    elif data_name == "METABRIC":
        return make_metabric()
    elif data_name == "NACD":
        return make_nacd()
    elif data_name == "FLCHAIN":
        return make_flchain()
    elif data_name == "GBSG2":
        return make_gbsg2()
    elif data_name == "NWTCO":
        return make_nwtco()
    elif data_name == "WHAS":
        return make_whas()
    elif data_name == "PBC":
        return make_pbc()
    elif data_name == "GBM":
        return make_gbm()
    elif data_name == "NPC":
        return make_npc()
    elif data_name == "AIDS":
        return make_aids()
    elif data_name == "HFCR":
        return make_heart_failure()
    elif data_name == "WPBC":
        return make_wpbc()
    elif data_name == "BMT":
        return make_bmt()
    elif data_name == "churn":
        return make_churn()
    elif data_name == "credit":
        return make_credit_risk()
    elif data_name == "employee":
        return make_employee_retention()
    elif data_name == "PDM":
        return make_pdm()
    elif data_name == "BCCardiotox":
        return make_BC_cardiotox()
    elif data_name == "kidney_transplant":
        return make_kidney_transplant()
    elif data_name == "larynx":
        return make_larynx()
    elif data_name == "leukemia":
        return make_leukemia()
    elif data_name == "NCCTG":
        return make_ncctg()
    elif data_name == "lupus":
        return make_lupus()
    elif data_name == "Rossi":
        return make_rossi()
    elif data_name == "MSKCC":
        return make_mskcc()
    elif data_name == "COVID":
        return make_covid()
    elif data_name == "MIMIC-IV_all":
        return make_mimic_iv_all()
    elif data_name == "SEER_brain":
        return make_seer_brain()
    elif data_name == "SEER_liver":
        return make_seer_liver()
    elif data_name in _SURVSET_DATA:
        # Load dataset from SurvSet
        loader = SurvLoader()
        data = loader.load_dataset(data_name)
        df = data["df"]
        df.rename(columns={"time": "time", "event": "event"}, inplace=True)
        # remove instances with non-positive survival time
        df = df[df["time"] > 0].reset_index(drop=True)
        # drop pid column if it exists
        if "pid" in df.columns:
            df.drop(columns=["pid"], inplace=True)
        # one-hot encode categorical variables, categorical variables start with 'fac_'
        cat_cols = [col for col in df.columns if col.startswith("fac_")]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df
    else:
        raise ValueError("Dataset name not recognized.")


def arff_to_dataframe(
    filename: str,
) -> pd.DataFrame:
    """
    Load an ARFF file and convert it to a pandas DataFrame.
    """
    data = scipy.io.arff.loadarff(filename)
    df = pd.DataFrame(data)
    return df


def add_prefix(
    df: pd.DataFrame,
    cont_list: list[str],
) -> pd.DataFrame:
    """
    Add a prefix ("num_") to the continuous columns in the DataFrame.
    """
    df = df.copy()
    for col in cont_list:
        if col in df.columns:
            df.rename(columns={col: f"num_{col}"}, inplace=True)
    return df


def make_support() -> pd.DataFrame:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://hbiostat.org/data/repo/supportdesc for more information.
    Download from https://hbiostat.org/data/repo/support2csv.zip
    Returns
    -------
    pd.DataFrame
        Processed covariates for one patient in each row.
    list[str]
        List of columns to standardize.

    References
    ----------
    [1] W. A. Knaus et al., The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    # url = "https://hbiostat.org/data/repo/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = [
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
        "sfdm2",
        "adlp",
        "adls",
        "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
        # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
        "sps",
        "aps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "hday",
    ]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    # df = pd.read_csv(url).drop(cols_to_drop, axis=1).rename(columns={"d.time": "time", "death": "event"})
    df = (
        pd.read_csv(f"{CURRENT_PATH}/support2.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"d.time": "time", "death": "event"})
    )
    df["event"] = df["event"].astype(int)

    df["ca"] = (df["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        # "edu": df["edu"].mean(),
        # "ph": df["ph"].mean(),
        # "glucose": df["glucose"].mean(),
        # "scoma": df["scoma"].mean(),
        # "meanbp": df["meanbp"].mean(),
        # "hrt": df["hrt"].mean(),
        # "resp": df["resp"].mean(),
        # "temp": df["temp"].mean(),
        # "sod": df["sod"].mean(),
        # "income": df["income"].mode()[0],
        # "race": df["race"].mode()[0],
    }
    df = df.fillna(fill_vals)

    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"male": 1, "female": 0}).infer_objects()
        df.income = df.income.replace(
            {"under $11k": 0, "$11-$25k": 1, "$25-$50k": 2, ">$50k": 3}
        ).infer_objects()
    skip_cols = [
        "event",
        "sex",
        "time",
        "dzclass",
        "race",
        "diabetes",
        "dementia",
        "ca",
    ]
    continuous_features = list(
        set(df.columns.to_list()).symmetric_difference(skip_cols)
    )

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    df = df.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    df.reset_index(drop=True, inplace=True)
    df = add_prefix(df, continuous_features)
    return df


def make_nacd() -> pd.DataFrame:
    cols_to_drop = ["PERFORMANCE_STATUS", "STAGE_NUMERICAL", "AGE65"]
    df = (
        pd.read_csv(f"{CURRENT_PATH}/NACD_Full.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"delta": "event"})
    )

    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    continuous_features = [
        "BOX1_SCORE",
        "BOX2_SCORE",
        "BOX3_SCORE",
        "BMI",
        "WEIGHT_CHANGEPOINT",
        "AGE",
        "GRANULOCYTES",
        "LDH_SERUM",
        "LYMPHOCYTES",
        "PLATELET",
        "WBC_COUNT",
        "CALCIUM_SERUM",
        "HGB",
        "CREATININE_SERUM",
        "ALBUMIN",
    ]
    df = add_prefix(df, continuous_features)
    return df


def make_metabric() -> pd.DataFrame:
    source = os.path.join(CURRENT_PATH, "Metabric.csv")
    if not os.path.exists(source):
        source = METABRIC_PAPER_URL
    df = pd.read_csv(source).rename(columns={"delta": "event", "duration": "time"})
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    continuous_features = [
        "age_at_diagnosis",
        "size",
        "lymph_nodes_positive",
        "stage",
        "lymph_nodes_removed",
        "NPI",
    ]
    df = add_prefix(df, continuous_features)
    return df



def make_flchain() -> pd.DataFrame:
    # flchain dataset: relationship between serum free light chain (FLC) and mortality
    # see: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html
    cols_to_drop = ["chapter"]  # only dead patients has chapter information
    df = (
        pd.read_csv(f"{CURRENT_PATH}/flchain.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"futime": "time", "death": "event"})
    )
    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"M": 1, "F": 0}).infer_objects()
    # processing see: https://github.com/paidamoyo/adversarial_time_to_event/blob/master/data/flchain/flchain_data.py
    # data = data.fillna({"creatinine": data["creatinine"].median()})
    onehot_cols = ["sample.yr", "flc.grp"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    skip_cols = {"event", "time", "sex", "mgus"}
    # assert not data.isnull().values.any(), "Dataset contains NaNs"
    continuous_features = list(
        set(df.columns.to_list()).symmetric_difference(skip_cols)
    )
    df = add_prefix(df, continuous_features)
    return df


def make_mimic_iv_all() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/MIMIC_IV_all_cause_failure.csv")
    skip_cols = [
        "event",
        "is_male",
        "time",
        "is_white",
        "renal",
        "cns",
        "coagulation",
        "cardiovascular",
    ]
    continuous_features = list(
        set(df.columns.to_list()).symmetric_difference(skip_cols)
    )
    df = add_prefix(df, continuous_features)
    return df



def make_nwtco() -> pd.DataFrame:
    """
    Tumor histology predicts survival. Downloaded and preprocessed from [1]_.

    Check the data description at https://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html
    Download from https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv

    References
    ----------
    [1] NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary outcome applied to
    Wilms tumour prognosis. Applied Statistics 48, 457–68.
    """
    cols_to_drop = ["rownames", "seqno"]
    df = (
        pd.read_csv(f"{CURRENT_PATH}/nwtco.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"rel": "event", "edrel": "time"})
    )

    df = add_prefix(df, ["age"])
    return df


def make_gbsg2() -> pd.DataFrame:
    """
    German Breast Cancer Study Group (GBSG)

    This dataset is downloaded from `survival` package in R.
    The data description can be found at https://rdrr.io/cran/survival/man/gbsg.html
    """
    cols_to_drop = ["pid"]
    df = (
        pd.read_csv(f"{CURRENT_PATH}/GBSG.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"status": "event", "rfstime": "time"})
    )

    continuous_features = ["age", "size", "grade", "nodes", "pgr", "er"]
    df = add_prefix(df, continuous_features)
    return df


def _make_df(data):
    x = data["x"]
    t = data["t"]
    d = data["e"]

    colnames = ["x" + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=colnames).assign(duration=t).assign(event=d)
    return df


def make_whas() -> pd.DataFrame:
    """
    Worcester Heart Attack Study dataset with 1638 samples and 6 covariates.
    Downloaded from https://github.com/sysucc-ailab/RankDeepSurv/tree/master/data/WHAS
    """
    df = defaultdict(dict)
    with h5py.File(f"{CURRENT_PATH}/whas_train_test.h5") as f:
        for ds in f:
            for array in f[ds]:
                df[ds][array] = f[ds][array][:]
    train = _make_df(df["train"])
    test = _make_df(df["test"])
    df = (
        pd.concat([train, test])
        .reset_index(drop=True)
        .rename(columns={"duration": "time"})
    )
    continuous_features = ["x1", "x3"]
    df = add_prefix(df, continuous_features)

    return df


def make_gbm() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/GBM.clin.merged.picked.csv").rename(
        columns={"delta": "event"}
    )
    df.drop(
        columns=["Composite Element REF", "tumor_tissue_site"], inplace=True
    )  # Columns with only one value
    df = df[df.time.notna()]  # Unknown censor/event time
    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    # Preprocess and fill missing values
    with pd.option_context("future.no_silent_downcasting", True):
        df.gender = df.gender.replace({"male": 1, "female": 0}).infer_objects()
        df.radiation_therapy = df.radiation_therapy.replace(
            {"yes": 1, "no": 0}
        ).infer_objects()
        df.ethnicity = df.ethnicity.replace(
            {"not hispanic or latino": 0, "hispanic or latino": 1}
        ).infer_objects()
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    # fill_vals = {
    #     "radiation_therapy": data["radiation_therapy"].median(),
    #     "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
    #     "ethnicity": data["ethnicity"].median()
    # }
    # data = data.fillna(fill_vals)
    df.columns = df.columns.str.replace(" ", "_")

    continuous_features = [
        "years_to_birth",
        "date_of_initial_pathologic_diagnosis",
        "karnofsky_performance_score",
    ]
    df = add_prefix(df, continuous_features)
    return df


def make_npc() -> pd.DataFrame:
    """
    nasopharyngeal carcinoma (NPC) prognostic dataset collected from
    Sun Yat-sen University Cancer Center, Guangzhou, China.

    End time is disease-free (Progression-free) survival time (PFSmonths), which was calculated
    from the date of diagnosis to the date of the first relapse at any site, death from any cause,
    or the date of the last follow-up visit.

    The original dataset are split into fixed training and testing set.
    The training set contains 4,630 consecutive NPC patients between 2007.01-2009.12.
    The testing set contains 1,819 NPC patients between 2011.01-2012.06.
    There we combine the training and testing set together.

    More details can be found in [1]_. The dataset is downloaded from [2]_.
    [1] Tang LQ, Li CF, Li J, et al. Establishment and Validation of Prognostic Nomograms
    for Endemic Nasopharyngeal Carcinoma. J Natl Cancer Inst. 2016. 108(1)
    [2] https://github.com/sysucc-ailab/RankDeepSurv/tree/master
    """
    df_train = pd.read_csv(f"{CURRENT_PATH}/npc_train.csv")
    df_test = pd.read_csv(f"{CURRENT_PATH}/npc_test.csv")
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    df.rename(
        columns={
            "PFSmonths": "time",
            "outcome": "event",
            "TUICC": "T_stage",
            "NUICC": "N_stage",
        },
        inplace=True,
    )

    continuous_features = ["CRP", "LDH", "age", "HGB", "BMI", "EBVDNA"]
    df = add_prefix(df, continuous_features)
    return df



def make_seer_liver() -> pd.DataFrame:
    """
    Preprocess the SEER liver cancer dataset.
    """
    df = pd.read_csv(f"{CURRENT_PATH}/SEER/Liver.csv").rename(
        columns={"Survival months": "time"}
    )
    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    skip_cols = ["event", "time", "Sex"]
    continuous_features = list(
        set(df.columns.to_list()).symmetric_difference(skip_cols)
    )
    df = add_prefix(df, continuous_features)

    return df




def make_seer_brain() -> pd.DataFrame:
    """
    Preprocess the SEER brain cancer dataset.
    """
    df = pd.read_csv(f"{CURRENT_PATH}/SEER/Brain.csv").rename(
        columns={"Survival months": "time"}
    )
    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    skip_cols = [
        "event",
        "time",
        "Sex",
        "Behavior recode for analysis",
        "SEER historic stage A (1973-2015)",
        "RX Summ--Scope Reg LN Sur (2003+)",
    ]
    continuous_features = list(
        set(df.columns.to_list()).symmetric_difference(skip_cols)
    )
    df = add_prefix(df, continuous_features)

    return df


def make_aids() -> pd.DataFrame:
    """
    Preprocess the AIDS Clinical Trials Group Study 175 dataset.

    Data link: https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
    Paper: https://www.nejm.org/doi/pdf/10.1056/NEJM199610103351501
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, "AIDS.csv"))


def make_pbc() -> pd.DataFrame:
    """
    Preprocess the Cirrhosis Patient Survival Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1
    Paper: https://pubmed.ncbi.nlm.nih.gov/2737595/
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, "PBC.csv"))


def make_heart_failure() -> pd.DataFrame:
    """
    Preprocess the Heart Failure Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
    Paper: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, "HFCR.csv"))


def make_wpbc() -> pd.DataFrame:
    """
    Preprocess the Wisconsin Prognostic Breast Cancer dataset.

    Event is recurrence of breast cancer.

    Link: https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, "WPBC.csv"))


def make_bmt() -> pd.DataFrame:
    """
    Preprocess for the Bone Marrow Transplant dataset.

    Link: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children
    Paper: https://www.astctjournal.org/article/S1083-8791(10)00148-5/fulltext
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, "BMT.csv"))

def make_churn() -> pd.DataFrame:
    """
    Predicting when your customers will churn.

    Data description: https://square.github.io/pysurvival/tutorials/churn.html
    Data downloaded from PySurvival: https://github.com/square/pysurvival/tree/master/pysurvival/datasets

    Link: https://www.kaggle.com/blastchar/telco-customer-churn
    """
    churn = pd.read_csv(f"{CURRENT_PATH}/churn.csv").rename(
        columns={"months_active": "time", "churned": "event"}
    )
    churn.event = churn.event.astype(int)

    with pd.option_context("future.no_silent_downcasting", True):
        churn.product_travel_expense = churn.product_travel_expense.replace(
            {"No": 0, "Free-Trial": 1, "Active": 2}
        ).infer_objects()
        churn.product_payroll = churn.product_payroll.replace(
            {"No": 0, "Free-Trial": 1, "Active": 2}
        ).infer_objects()
        churn.product_accounting = churn.product_accounting.replace(
            {"No": 0, "Free-Trial": 1, "Active": 2}
        ).infer_objects()
        churn.company_size = churn.company_size.replace(
            {
                "self-employed": 0,
                "1-10": 1,
                "10-50": 2,
                "50-100": 3,
                "100-250": 4,
            }
        ).infer_objects()
    # creating one-hot vectors for categorical variables
    cat_cols = ["us_region"]
    df = pd.get_dummies(churn, columns=cat_cols, drop_first=True)

    df = df.drop(
        df[df["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    continuous_features = [
        "product_data_storage",
        "product_travel_expense",
        "product_payroll",
        "product_accounting",
        "csat_score",
        "articles_viewed",
        "marketing_emails_clicked",
        "minutes_customer_support",
        "company_size",
    ]
    df = add_prefix(df, continuous_features)

    return df


def make_credit_risk() -> pd.DataFrame:
    """
    Predicting credit risk.

    Data description:
        https://square.github.io/pysurvival/tutorials/credit_risk.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/credit_risk.csv
    """
    credit_risk = pd.read_csv(f"{CURRENT_PATH}/credit_risk.csv").rename(
        columns={"duration": "time", "full_repaid": "event"}
    )

    with pd.option_context("future.no_silent_downcasting", True):
        credit_risk.checking_account_status = (
            credit_risk.checking_account_status.replace(
                {"no_account": 0, "below_0": -1, "0_to_200": 1, "above_200": 2}
            ).infer_objects()
        )
        credit_risk.credit_history = credit_risk.credit_history.replace(
            {
                "all_credit_paid": 2,
                "existing_credit_paid": 1,
                "no_credit_taken": 0,
                "delay_in_paying": -1,
                "critical_account": -2,
            }
        ).infer_objects()
        credit_risk.savings_account_status = credit_risk.savings_account_status.replace(
            {
                "unknown": np.nan,
                "below_100": 0,
                "between_100_500": 1,
                "between_500_1000": 2,
                "above_1000": 3,
            }
        ).infer_objects()
        credit_risk.employment_years = credit_risk.employment_years.replace(
            {
                "unemployed": 0,
                "below_1": 1,
                "between_1_4": 2,
                "between_4_7": 3,
                "above_7": 4,
            }
        ).infer_objects()
        credit_risk.job = credit_risk.job.replace(
            {"unemployed": 0, "unskilled": 1, "official": 2, "management": 3}
        ).infer_objects()
    # creating one-hot vectors for categorical variables
    cat_cols = [
        "purpose",
        "personal_status",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
    ]
    df = pd.get_dummies(credit_risk, columns=cat_cols, drop_first=True)

    # fill the missing values with the median
    # data = data.fillna(data.median())

    continuous_features = [
        "checking_account_status",
        "credit_history",
        "amount",
        "savings_account_status",
        "employment_years",
        "installment_rate",
        "present_residence",
        "age",
        "number_of_credits",
        "job",
        "people_liable",
    ]
    df = add_prefix(df, continuous_features)

    return df


def make_employee_retention() -> pd.DataFrame:
    """
    Predicting employee retention.

    Data description:
        https://square.github.io/pysurvival/tutorials/employee_retention.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/employee_attrition.csv
    """
    retention = pd.read_csv(f"{CURRENT_PATH}/employee_attrition.csv").rename(
        columns={"time_spend_company": "time", "left": "event"}
    )

    with pd.option_context("future.no_silent_downcasting", True):
        retention.salary = retention.salary.replace(
            {"low": 0, "medium": 1, "high": 2}
        ).infer_objects()
    # creating one-hot vectors for categorical variables
    cat_cols = ["department"]
    df = pd.get_dummies(retention, columns=cat_cols, drop_first=True)

    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    continuous_features = [
        "satisfaction_level",
        "last_evaluation",
        "number_projects",
        "average_montly_hours",
        "work_accident",
        "promotion_last_5years",
        "salary",
    ]

    df = add_prefix(df, continuous_features)

    return df


def make_pdm() -> pd.DataFrame:
    """
    Predictive maintenance dataset.

    Data description:
        https://square.github.io/pysurvival/tutorials/maintenance.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/maintenance.csv
    """
    pdm = pd.read_csv(f"{CURRENT_PATH}/maintenance.csv", sep=";").rename(
        columns={"lifetime": "time", "broken": "event"}
    )

    cat_cols = ["team", "provider"]
    df = pd.get_dummies(pdm, columns=cat_cols, drop_first=True)

    continuous_features = [
        "pressureInd",
        "moistureInd",
        "temperatureInd",
    ]

    df = add_prefix(df, continuous_features)

    return df


def make_BC_cardiotox() -> pd.DataFrame:
    """
    BC_cardiotox database, from papr: https://www.nature.com/articles/s41597-023-02419-1
    The data can be downloaded from
    https://figshare.com/articles/dataset/BC_cardiotox_A_cardiotoxicity_dataset_for_breast_cancer_patients/22650748

    It contains information about 531 breast cancer patients and aims to enable the scientific community in
    conducting new research on cancer therapy-related cardiac dysfunction (CTRCD).

    Here we only use the clinical variables, not the functional image variable.
    """
    bc_cardiotox = pd.read_csv(
        f"{CURRENT_PATH}/BC_cardiotox_clinical_variables.csv", sep=";"
    ).rename(columns={"CTRCD": "event"})

    # in the feature 'height', it uses ',' as decimal separator, so we need to replace it with '.'
    features_need_replace = ["height", "LVEF", "PWT", "LAd", "LVDd", "LVSd"]
    for feature in features_need_replace:
        bc_cardiotox[feature] = (
            bc_cardiotox[feature].str.replace(",", ".").astype(float)
        )

    continuous_features = [
        "heart_rate",
        "age",
        "weight",
        "height",
        "heart_rhythm",
        "LVEF",
        "PWT",
        "LAd",
        "LVDd",
        "LVSd",
    ]
    df = add_prefix(bc_cardiotox, continuous_features)

    return df


def make_kidney_transplant() -> pd.DataFrame:
    """
    Kidney Transplant Survival dataset.

    D.3 from Klein and Moeschberger Statistics for Biology and Health, 1997.
    """
    kidney_transplant = pd.read_csv(f"{CURRENT_PATH}/kidney_transplant.csv").rename(
        columns={"death": "event"}
    )

    df = add_prefix(kidney_transplant, ["age"])
    return df


def make_larynx() -> pd.DataFrame:
    """
    Larynx Cancer dataset.

    From lifelines  package.
    """
    larynx = pd.read_csv(f"{CURRENT_PATH}/larynx.csv").rename(
        columns={"death": "event"}
    )

    df = add_prefix(larynx, ["age"])
    return df


def make_leukemia() -> pd.DataFrame:
    """
    Leukemia dataset.
    From lifelines package.
    which is originally from
    http://web1.sph.emory.edu/dkleinb/allDatasets/surv2datasets/anderson.dat
    """
    leukemia = pd.read_csv(f"{CURRENT_PATH}/anderson.csv", sep=" ").rename(
        columns={"status": "event", "t": "time"}
    )

    df = add_prefix(leukemia, ["logWBC"])
    return df


def make_ncctg() -> pd.DataFrame:
    """
    Survival in patients with advanced lung cancer from the North Central Cancer Treatment Group. Performance scores rate how well the patient can perform usual daily activities.

    Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ. Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al.
    Prospective evaluation of prognostic variables from patient-completed questionnaires.
    North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.
    """
    ncctg = pd.read_csv(f"{CURRENT_PATH}/lung.csv").rename(columns={"status": "event"})

    # sex are 1 and 2, change to 0 and 1
    ncctg.sex = ncctg.sex - 1

    # ph.ecog refers to the Eastern Cooperative Oncology Group (ECOG) performance score.
    continuous_features = [
        "inst",
        "age",
        "ph.ecog",
        "ph.karno",
        "pat.karno",
        "meal.cal",
        "wt.loss",
    ]
    df = add_prefix(ncctg, continuous_features)
    return df


def make_lupus() -> pd.DataFrame:
    """
    Survival times of 98 lupus erythematosus patients.

    The dataset is translated from the original paper by the author of lifelines package.

    https://projecteuclid.org/download/pdf_1/euclid.aos/1176345693
    Merrell, M., & Shulman, L. E. (1955). Determination of prognosis in chronic disease,
    illustrated by systemic lupus erythematosus. Journal of Chronic Diseases, 1(1), 12–32.
    doi:10.1016/0021-9681(55)90018-7
    """
    col_to_drop = ["id", "year_month_of_last_observation"]
    lupus = (
        pd.read_csv(f"{CURRENT_PATH}/merrell1955.csv")
        .drop(col_to_drop, axis=1)
        .rename(
            columns={
                "time_between_diagnosis_and_last_observation_(years)": "time",
                "dead": "event",
            }
        )
    )
    lupus.event = lupus.event.astype(int)
    lupus = lupus.drop(
        lupus[lupus["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    lupus.reset_index(drop=True, inplace=True)

    # change "year_month_of_diagnosis" to relative time in months, not year-month format
    # set the earliest diagnosis time to 0, so that the time is relative to the earliest diagnosis
    lupus["recency_of_diagnosis"] = (
        pd.to_datetime(lupus["year_month_of_diagnosis"], format="%Y-%m")
        - pd.to_datetime(lupus["year_month_of_diagnosis"].min(), format="%Y-%m")
    ).dt.days / 30
    lupus["recency_of_diagnosis"] = np.round(lupus["recency_of_diagnosis"]).astype(int)
    lupus = lupus.drop(columns=["year_month_of_diagnosis"])
    continuous_features = [
        "recency_of_diagnosis",
        "age_at_diagnosis",
        "time_elapsed_between_estimated_onset_and_diagnosis_(months)",
    ]
    df = add_prefix(lupus, continuous_features)
    return df


def make_rossi() -> pd.DataFrame:
    """
    The Rossi dataset pertain to 432 convicts who were released from Maryland state prisons in the 1970s
    and who were followed up for one year after release.
    Half the released convicts were assigned at random to an experimental treatment in which
    they were given financial aid; half did not receive aid.

    Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime: Some Experimental Results.
    New York: Academic Press.

    John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander
    Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.
    """
    rossi = pd.read_csv(f"{CURRENT_PATH}/rossi.csv").rename(
        columns={"week": "time", "arrest": "event"}
    )
    df = add_prefix(rossi, ["age", "prio"])
    return df


def remove_auxiliary_categories(df, feature, threshold=10):
    """
    Remove auxiliary categories from a categorical feature in a DataFrame.

    If a category has less than `threshold` samples, it will be replaced with NaN.
    Then when one-hot encoding is applied, these categories will be dropped.
    """
    df[feature] = df[feature].replace(
        df[feature].value_counts()[df[feature].value_counts() < threshold].index, np.nan
    )
    return df


def make_mskcc():
    """
    MSK-IMPACT Clinical Sequencing Cohort (MSKCC, Nat Med 2017)
    See paper: https://pubmed.ncbi.nlm.nih.gov/28481359/
    """
    cols_drop = [
        "Study ID",
        "Patient ID",
        "Sample ID",
        "Cancer Type Detailed",
        "Sample Class",
        "Patient's Vital Status",
    ]
    msk = (
        pd.read_csv(f"{CURRENT_PATH}/msk_impact_2017_clinical_data.tsv", sep="\t")
        .drop(cols_drop, axis=1)
        .rename(
            columns={
                "Overall Survival (Months)": "time",
                "Overall Survival Status": "event",
            }
        )
    )

    # drop rows with nan in 'time' or 'event'
    msk = msk.dropna(subset=["time", "event"])
    msk = msk.reset_index(drop=True)
    msk["event"] = msk["event"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(int)

    # drop rows with negative or zero survival time
    msk = msk[msk["time"] > 0].reset_index(drop=True)

    msk = remove_auxiliary_categories(msk, "Cancer Type", threshold=10)
    msk = remove_auxiliary_categories(msk, "Metastatic Site", threshold=20)
    msk = remove_auxiliary_categories(msk, "Oncotree Code", threshold=20)
    msk = remove_auxiliary_categories(msk, "Primary Tumor Site", threshold=10)
    msk = remove_auxiliary_categories(msk, "Specimen Preservation Type", threshold=10)
    msk = remove_auxiliary_categories(msk, "Specimen Type", threshold=2)
    # for Smoking History, change 'Unknown' to NaN
    msk["Smoking History"] = msk["Smoking History"].replace({"Unknown": np.nan})

    category_features = [
        "Cancer Type",
        "Matched Status",
        "Metastatic Site",
        "Oncotree Code",
        "Primary Tumor Site",
        "Sample Collection Source",
        "Sample Type",
        "Sex",
        "Smoking History",
        "Somatic Status",
        "Specimen Preservation Type",
        "Specimen Type",
    ]
    labels = ["time", "event"]
    continuous_features = [
        col for col in msk.columns if col not in category_features + labels
    ]

    msk = pd.get_dummies(msk, columns=category_features, drop_first=True)
    df = add_prefix(msk, continuous_features)
    return df


def make_covid():
    """
    Load the COVID-19 dataset.

    This dataset aims to investigate the discharge time of COVID-19 patients in Asian.

    Data link: https://github.com/kuan0911/ISDEvaluation-covid/blob/master/Data/covid/asian_discharge_exp3.csv
    Paper: https://www.nature.com/articles/s41598-022-08601-6#MOESM1
    """
    covid = pd.read_csv(f"{CURRENT_PATH}/asian_discharge_exp3.csv")

    covid = covid.drop(
        covid[covid["time"] <= 0].index
    )  # remove patients with negative or zero survival time
    covid.reset_index(drop=True, inplace=True)

    # change population density to float
    covid["population_density_city"] = (
        covid["population_density_city"].str.replace(",", "").astype(float)
    )
    continuous_features = [
        "age",
        "latitude",
        "longitude",
        "population_density_city",
        "population_density_country",
        "GDP_per_capita_country",
        "GDP_total_country",
    ]
    df = add_prefix(covid, continuous_features)
    return df
