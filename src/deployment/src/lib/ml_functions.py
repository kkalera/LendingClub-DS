import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from .data_functions import get_one_hot_encoded
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin

from sklearn.metrics import roc_auc_score, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import optuna
import os
import joblib


def run_pca(data, n_components=2):
    imputer = SimpleImputer(strategy="most_frequent")
    pca = PCA(n_components=n_components)

    data_oh = get_one_hot_encoded(data)
    data_imp = imputer.fit_transform(data_oh)
    pca.fit(data_imp)

    results = pd.DataFrame(
        pca.components_.T,
        index=data_oh.columns,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )

    return list(set([results[col].idxmax() for col in results.columns]))


def get_pipeline(
    model,
    categorical_steps=[],
    numerical_steps=[],
    use_SMOTE=False,
):
    """
    Create a data preprocessing pipeline for a given model.

    Parameters:
    model (object): The machine learning model to be used in the pipeline.

    Returns:
    pipeline (object): The data preprocessing pipeline.

    """
    transformers = []

    if categorical_columns is not None:
        categorical_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="error")),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    # Do standard scaling for the numerical columns
    if numerical_columns is not None:
        numerical_transformer = Pipeline(
            steps=[
                # ("impute", KNNImputer()), Disabled for performance reasons
                ("impute", SimpleImputer(strategy="mean")),
                ("IQR scaler", IQRScaler(factor=1.5)),
                ("scale", StandardScaler()),
                ("normalize", Normalizer()),
            ]
        )
        transformers.append(("numerical", numerical_transformer, numerical_columns))

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
    )

    # Create the pipeline
    if use_SMOTE:
        pipeline = Imb_pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ]
        )
    else:
        pipeline = Imb_pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
    return pipeline


def train_test_valid_split(
    data,
    label_column,
    stratify: str = None,
    test_size=0.2,
    valid_size=0.2,
    random_state=42,
):
    """
    Splits the data into training, validation and test sets.

    Parameters:
    - data: The input data.
    - label_column: The name of the label column.
    - stratify: The name of the column to use for stratification.
    - test_size: The size of the test set.
    - valid_size: The size of the validation set.
    - random_state: The random state for reproducibility.

    Returns:
    - train: The training set.
    - valid: The validation set.
    - test: The test set.
    """
    # Split into train and test set
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[stratify] if stratify else None,
    )

    # Split train set into train and validation set
    train, valid = train_test_split(
        train,
        test_size=valid_size,
        random_state=random_state,
        stratify=train[stratify] if stratify else None,
    )

    # Split into input features and target labels
    x_train = train.drop(columns=label_column)
    y_train = train[label_column]
    x_valid = valid.drop(columns=label_column)
    y_valid = valid[label_column]
    x_test = test.drop(columns=label_column)
    y_test = test[label_column]

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_valid": x_valid,
        "y_valid": y_valid,
        "X_test": x_test,
        "y_test": y_test,
    }


def xgb_classification_objective(
    trial,
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_steps,
    cat_steps,
    enable_categorical=False,
    use_class_weights=False,
    use_scale_pos_weight=False,
    use_smote=False,
):

    if use_class_weights:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[y] for y in y_train])

    if use_scale_pos_weight:
        vc = y_train.value_counts(normalize=True)
        scale_pos_weight = vc[0] / vc[1]
        if enable_categorical:
            model = get_xgb_classifier(
                trial, scale_pos_weight=scale_pos_weight, enable_categorical=True
            )
        else:
            model = get_xgb_classifier(trial, scale_pos_weight=scale_pos_weight)
    else:
        if enable_categorical:
            model = get_xgb_classifier(trial, enable_categorical=True)
        else:
            model = get_xgb_classifier(trial)

    pipeline = PipeBuilder().build(
        model=model,
        cat_columns=X_train.select_dtypes(include=["object", "category"]).columns,
        num_columns=X_train.select_dtypes(include=np.number).columns,
        cat_steps=cat_steps,
        num_steps=num_steps,
        use_smote=use_smote,
    )

    if use_class_weights:
        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights,
        )
    else:
        pipeline.fit(X_train, y_train)

    return roc_auc_score(y_valid, pipeline.predict_proba(X_valid)[:, 1])


def get_xgb_classifier(
    trial, objective="binary:logistic", scale_pos_weight=1, enable_categorical=False
) -> XGBClassifier:
    """
    Based upon: https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451

    Instantiate a XGBClassifier model with the given hyperparameters.

    Parameters:
    - trial (optuna.trial): The optuna trial object containing the hyperparameters.

    Returns:
    - model (XGBClassifier): The instantiated XGBClassifier model.
    """

    metric_list = ["logloss", "auc", "error"]
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        # "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
        "eval_metric": "auc",
        "objective": objective,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.01),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1, step=0.01),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1, step=0.01),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "scale_pos_weight": scale_pos_weight,
    }
    if enable_categorical:
        return XGBClassifier(**params, enable_categorical=True)
    else:
        return XGBClassifier(**params)


def evaluate_model_value(pipe, X_test, y_test):
    df = pd.concat([X_test, y_test], axis=1)
    df["pred"] = pipe.predict(X_test)

    tpr = df.loc[(df["TARGET"] == 1) & (df["pred"] == 1)]["AMT_CREDIT"].sum()
    fpr = df.loc[(df["TARGET"] == 0) & (df["pred"] == 1)]["AMT_CREDIT"].sum()
    tnr = df.loc[(df["TARGET"] == 0) & (df["pred"] == 0)]["AMT_CREDIT"].sum()
    fnr = df.loc[(df["TARGET"] == 1) & (df["pred"] == 0)]["AMT_CREDIT"].sum()

    model_value = (tpr + tnr) - (fpr + fnr)
    current_value = (
        df.loc[df["TARGET"] == 0]["AMT_CREDIT"].sum()
        - df.loc[df["TARGET"] == 1]["AMT_CREDIT"].sum()
    )
    results = pd.DataFrame(
        {
            "value": [current_value, model_value],
            "source": ["current", "model"],
        }
    )
    return results


def segmentation_model(path, data):
    if not os.path.exists(path):
        builder = PipeBuilder()
        pipe = builder.build(
            model=KMeans(n_clusters=5, random_state=42, n_init=10),
            num_columns=data.drop(columns=["TARGET"])
            .select_dtypes(include="number")
            .columns,
            cat_columns=data.drop(columns=["TARGET"])
            .select_dtypes(include=["category", "object"])
            .columns,
            num_steps=[
                builder.steps["simple_imputer_median"],
                builder.steps["iqr_scaler"],
                builder.steps["standard_scaler"],
                builder.steps["normalizer"],
            ],
            cat_steps=[
                builder.steps["simple_imputer_most_frequent"],
                builder.steps["one_hot_encoder"],
            ],
        )

        pipe.fit(data.drop(columns=["TARGET"]))

        joblib.dump(pipe, path)
        return pipe
    else:
        return joblib.load(path)


def approval_model_0(path, data):
    builder = PipeBuilder()

    num_steps = [
        builder.steps["simple_imputer_median"],
        builder.steps["iqr_scaler"],
        builder.steps["standard_scaler"],
        builder.steps["normalizer"],
    ]

    cat_steps = [
        builder.steps["simple_imputer_most_frequent"],
        builder.steps["one_hot_encoder"],
    ]

    if not os.path.exists(path):

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: xgb_classification_objective(
                trial,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_valid=data["X_valid"],
                y_valid=data["y_valid"],
                num_steps=num_steps,
                cat_steps=cat_steps,
                use_class_weights=True,
                use_scale_pos_weight=True,
            ),
            n_trials=25,
        )

        pipe = builder.build(
            model=XGBClassifier(**study.best_params),
            num_columns=data["X_train"].select_dtypes(include=np.number).columns,
            cat_columns=data["X_train"]
            .select_dtypes(include=["category", "object"])
            .columns,
            num_steps=num_steps,
            cat_steps=cat_steps,
        ).fit(data["X_train"], data["y_train"])

        joblib.dump(pipe, path)
        return pipe

    else:

        return joblib.load(path)


def approval_model_1(path, data):
    builder = PipeBuilder()

    num_steps = [
        builder.steps["simple_imputer_median"],
        builder.steps["iqr_scaler"],
        builder.steps["standard_scaler"],
        builder.steps["normalizer"],
    ]

    cat_steps = [
        builder.steps["simple_imputer_most_frequent"],
        builder.steps["one_hot_encoder"],
    ]

    if not os.path.exists(path):

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: xgb_classification_objective(
                trial,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_valid=data["X_valid"],
                y_valid=data["y_valid"],
                num_steps=num_steps,
                cat_steps=cat_steps,
                use_smote=True,
            ),
            n_trials=25,
        )

        pipe = builder.build(
            model=XGBClassifier(**study.best_params),
            num_columns=data["X_train"].select_dtypes(include=np.number).columns,
            cat_columns=data["X_train"]
            .select_dtypes(include=["category", "object"])
            .columns,
            num_steps=num_steps,
            cat_steps=cat_steps,
            use_smote=True,
        ).fit(data["X_train"], data["y_train"])

        joblib.dump(pipe, path)
        return pipe

    else:

        return joblib.load(path)


def run_rfecv(pipe, X_train, y_train, path):
    if os.path.exists(path):
        rfecv = joblib.load(path)
    else:
        rfecv = RFECV(
            estimator=pipe["model"],
            step=1,
            scoring="roc_auc",
            n_jobs=-1,
        )
        rfecv.fit(pipe.named_steps["preprocessor"].transform(X_train), y_train)
        joblib.dump(rfecv, path)

    encoded_features = np.array(
        pipe.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out()
        .tolist()
        + X_train.select_dtypes(include="number").columns.tolist()
    )

    chosen_features = encoded_features[rfecv.support_]
    return find_represented_original_columns(chosen_features, X_train.columns)


def find_represented_original_columns(mixed_columns, original_columns):
    """
    Identifies which original columns are represented in the mixed list of columns.

    Parameters:
    - mixed_columns: List of strings, containing mixed one-hot encoded and non-encoded column names.
    - original_columns: List of strings, containing the original column names before any encoding.

    Returns:
    - List of original column names that are represented in the mixed columns list.
    """
    represented_columns = []

    for original_col in original_columns:
        # Check if the original column is directly in the mixed list (non-encoded case)
        if original_col in mixed_columns:
            represented_columns.append(original_col)
        else:
            # Check for one-hot encoded representations of the original column
            for mixed_col in mixed_columns:
                if mixed_col.startswith(original_col + "_"):
                    represented_columns.append(original_col)
                    break  # Found a representation, no need to check further for this column

    return represented_columns


def get_feature_names(pipe, X):
    encoded_features = (
        pipe.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out()
        .tolist()
        + X.select_dtypes(include="number").columns.tolist()
    )

    return encoded_features


class IQRScaler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * IQR
        self.upper_bound = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound, self.upper_bound)


class KMeansPyTorch(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, distance="euclidean", options=None):
        self.n_clusters = n_clusters
        self.distance = distance
        self.options = options if options is not None else {}

    def fit(self, X, y=None):
        # Convert X to a PyTorch tensor
        data_tensor = torch.tensor(X.astype(np.float32))
        # Perform KMeans clustering
        cluster_ids_x, cluster_centers = kmeans(
            X=data_tensor,
            num_clusters=self.n_clusters,
            distance=self.distance,
            device=torch.device("cuda:0"),
            **self.options,
        )
        self.cluster_centers_ = cluster_centers
        self.labels_ = cluster_ids_x
        return self

    def predict(self, X):
        # Convert X to a PyTorch tensor
        data_tensor = torch.tensor(X.astype(np.float32))
        # Compute distances to cluster centers
        distances = torch.cdist(data_tensor, self.cluster_centers_)
        # Return the index of the closest cluster center
        return torch.argmin(distances, dim=1).numpy()


class PipeBuilder:

    from sklearn.preprocessing import (
        StandardScaler,
        OneHotEncoder,
        OrdinalEncoder,
        Normalizer,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from imblearn.pipeline import Pipeline as Imb_pipeline
    from imblearn.over_sampling import SMOTE

    def __init__(self):
        self.steps = {
            "simple_imputer_most_frequent": (
                "simple_imputer_most_frequent",
                SimpleImputer(strategy="most_frequent"),
            ),
            "simple_imputer_mean": (
                "simple_imputer_mean",
                SimpleImputer(strategy="mean"),
            ),
            "simple_imputer_median": (
                "simple_imputer_median",
                SimpleImputer(strategy="median"),
            ),
            "standard_scaler": ("standard_scaler", StandardScaler()),
            "iqr_scaler": ("iqr_scaler", IQRScaler()),
            "one_hot_encoder": (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
            "ordinal_encoder": ("ordinal_encoder", OrdinalEncoder()),
            "normalizer": ("normalizer", Normalizer()),
        }

    def build(
        self,
        model,
        cat_columns=[],
        cat_steps=[],
        num_columns=[],
        num_steps=[],
        use_smote=False,
    ):
        transformers = []

        if len(cat_steps) > 0:
            transformers.append(
                ("categorical_preprocessor", Pipeline(steps=cat_steps), cat_columns)
            )

        if len(num_steps) > 0:
            transformers.append(
                ("numerical_preprocessor", Pipeline(steps=num_steps), num_columns)
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        if use_smote:
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("model", model),
                ]
            )
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

        return pipeline
