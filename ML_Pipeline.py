import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
    KFold,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import joblib

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred, **kwargs):
        return np.sqrt(mean_squared_error(y_true, y_pred, **kwargs))

warnings.filterwarnings("ignore")

INPUT_FILE   = "datasetname_features.csv"
OUTDIR       = "Results_datasetname_MLM"
RANDOM_STATE = 42
N_JOBS       = 4

os.makedirs(OUTDIR, exist_ok=True)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, y_min=0.0):
        self.estimator = estimator
        self.y_min = y_min

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.estimator.predict(X), self.y_min, None)

    def get_params(self, deep=True):
        return {"estimator": self.estimator, "y_min": self.y_min}


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def evaluate_cv(name, estimator, X_tr, y_tr, cv_splitter, groups_tr=None, split_label="grouped"):
    fold_metrics = {"fold": [], "mae": [], "rmse": [], "r2": []}

    if isinstance(cv_splitter, GroupKFold) and groups_tr is not None:
        splitter = cv_splitter.split(X_tr, y_tr, groups=groups_tr)
    else:
        splitter = cv_splitter.split(X_tr, y_tr)

    for fold, (tr, val) in enumerate(splitter, 1):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_tr.iloc[tr], y_tr[tr])
        y_pred = fold_estimator.predict(X_tr.iloc[val])
        mae, rmse, r2 = compute_metrics(y_tr[val], y_pred)
        fold_metrics["fold"].append(fold)
        fold_metrics["mae"].append(mae)
        fold_metrics["rmse"].append(rmse)
        fold_metrics["r2"].append(r2)

    mean_mae = np.mean(fold_metrics["mae"])
    std_mae = np.std(fold_metrics["mae"])
    mean_rmse = np.mean(fold_metrics["rmse"])
    std_rmse = np.std(fold_metrics["rmse"])
    mean_r2 = np.mean(fold_metrics["r2"])
    std_r2 = np.std(fold_metrics["r2"])

    return {
        "model": name, "split": split_label,
        "mean_mae": mean_mae, "std_mae": std_mae,
        "mean_rmse": mean_rmse, "std_rmse": std_rmse,
        "mean_r2": mean_r2, "std_r2": std_r2,
        "fold_metrics": fold_metrics,
    }


def run_cv_and_test(X, y, models, split_mode, groups=None, test_size=0.2, random_state=RANDOM_STATE):
    assert split_mode in ("group", "random")

    if split_mode == "group":
        if groups is None:
            raise ValueError("groups must be provided for split_mode='group'")
        outer_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(outer_split.split(X, y, groups=groups))
        split_label = "grouped_chromophore"
    else:
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
        split_label = "random_split"

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    if split_mode == "group":
        groups_train = groups[train_idx]
        inner_cv = GroupKFold(n_splits=5)
    else:
        groups_train = None
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_results = []
    for name, model in models.items():
        result = evaluate_cv(name, model, X_train, y_train, cv_splitter=inner_cv,
                              groups_tr=groups_train, split_label=split_label)
        cv_results.append(result)

    test_results = []
    for name, model in models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred_test = m.predict(X_test)
        mae_test, rmse_test, r2_test = compute_metrics(y_test, y_pred_test)
        test_results.append({
            "model": name, "split": split_label,
            "mae": mae_test, "rmse": rmse_test, "r2": r2_test,
            "y_true": y_test, "y_pred": y_pred_test,
        })

    return {
        "split_mode": split_mode, "split_label": split_label,
        "train_idx": train_idx, "test_idx": test_idx,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "groups_train": groups_train,
        "cv_results": cv_results, "test_results": test_results,
    }


def load_params(csv_path):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    params = df.iloc[:, 0].to_dict()
    int_keys = ["n_estimators", "num_leaves", "max_depth", "min_child_samples",
                "min_child_weight", "reg_alpha", "reg_lambda"]
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
    return params


def eval_model(model, X_train, y_train, X_test, y_test, model_name, split_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, rmse, r2 = compute_metrics(y_test, y_pred)
    return {
        "model": model_name,
        "split": split_label,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_test,
        "y_pred": y_pred,
        "estimator": model,
        "best_params": model.get_params(),
    }


def shap_and_importance_for_model(estimator, X_train, feature_cols, out_prefix):
    sample_size = min(1000, len(X_train))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_sample)

    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Feature": list(X_sample.columns), "Importance": feature_importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.head(30).to_csv(os.path.join(OUTDIR, f"{out_prefix}_feature_importance_top30.csv"), index=False)

    return importance_df, shap_values, X_sample


df = pd.read_csv(INPUT_FILE)
y = df["Stokes Shift (nm)"].values

exclude_cols = ["Chromophore", "Solvent", "Reference",
                "Absorption max (nm)", "Emission max (nm)", "Stokes Shift (nm)"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

nan_counts = df[feature_cols].isnull().sum()
cols_with_nan = nan_counts[nan_counts > 0]

if len(cols_with_nan) > 0:
    nan_cols_all = [c for c in feature_cols if df[feature_cols][c].isnull().all()]
    if nan_cols_all:
        feature_cols = [c for c in feature_cols if c not in nan_cols_all]

    threshold = 0.5
    cols_to_drop = [c for c in cols_with_nan.index if cols_with_nan[c] / len(df) > threshold]
    if cols_to_drop:
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df[feature_cols]), columns=feature_cols)
    joblib.dump(imputer, os.path.join(OUTDIR, "median_imputer.joblib"))

    if np.isnan(X.values).sum() != 0:
        raise ValueError("NaN values remain after imputation — check input data.")
else:
    X = df[feature_cols].copy()

solvent_features = [c for c in feature_cols if c.startswith("solv_")]
chrom_features = [c for c in feature_cols if c.startswith("desc_")]
morgan_features = [c for c in feature_cols if c.startswith("morgan_")]
maccs_features = [c for c in feature_cols if c.startswith("maccs_")]

groups_chrom = df["Chromophore"].values

pd.DataFrame({"Feature": feature_cols}).to_csv(os.path.join(OUTDIR, "feature_names.csv"), index=False)

models = {
    "Ridge": ClippedRegressor(
        Pipeline([("scaler", RobustScaler()), ("model", Ridge(alpha=1.0))]), y_min=1.0
    ),
    "Lasso": ClippedRegressor(
        Pipeline([("scaler", RobustScaler()), ("model", Lasso(alpha=0.1, max_iter=10000))]), y_min=1.0
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", n_jobs=N_JOBS, random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, gamma=0,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500, max_depth=-1, learning_rate=0.1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
    ),
}

group_results = run_cv_and_test(X, y, models, split_mode="group", groups=groups_chrom)
random_results = run_cv_and_test(X, y, models, split_mode="random", groups=None)

cv_results_all = group_results["cv_results"] + random_results["cv_results"]
test_results_all = group_results["test_results"] + random_results["test_results"]

tuned_group_lgbm = tune_lightgbm_for_split(
    group_results["X_train"], group_results["y_train"],
    group_results["X_test"], group_results["y_test"],
    groups_train=group_results["groups_train"],
    split_label=group_results["split_label"], n_iter_search=50,
)

tuned_random_lgbm = tune_lightgbm_for_split(
    random_results["X_train"], random_results["y_train"],
    random_results["X_test"], random_results["y_test"],
    groups_train=None, split_label=random_results["split_label"], n_iter_search=50,
)

tuned_group_xgb = tune_xgboost_for_split(
    group_results["X_train"], group_results["y_train"],
    group_results["X_test"], group_results["y_test"],
    groups_train=group_results["groups_train"],
    split_label=group_results["split_label"], n_iter_search=50,
)

tuned_random_xgb = tune_xgboost_for_split(
    random_results["X_train"], random_results["y_train"],
    random_results["X_test"], random_results["y_test"],
    groups_train=None, split_label=random_results["split_label"], n_iter_search=50,
)

for tuned in [tuned_group_lgbm, tuned_random_lgbm, tuned_group_xgb, tuned_random_xgb]:
    test_results_all.append({
        "model": tuned["model"],
        "split": tuned["split"],
        "mae": tuned["mae"],
        "rmse": tuned["rmse"],
        "r2": tuned["r2"],
        "y_true": tuned["y_true"],
        "y_pred": tuned["y_pred"],
    })

cv_df = pd.DataFrame([{
    "Model": r["model"], "Split": r["split"],
    "CV_MAE_mean": r["mean_mae"], "CV_MAE_std": r["std_mae"],
    "CV_RMSE_mean": r["mean_rmse"], "CV_RMSE_std": r["std_rmse"],
    "CV_R2_mean": r["mean_r2"], "CV_R2_std": r["std_r2"],
} for r in cv_results_all])
cv_df.to_csv(os.path.join(OUTDIR, "cv_results_all.csv"), index=False)

test_df = pd.DataFrame([{
    "Model": r["model"], "Split": r["split"],
    "Test_MAE": r["mae"], "Test_RMSE": r["rmse"], "Test_R2": r["r2"],
} for r in test_results_all])
test_df.to_csv(os.path.join(OUTDIR, "test_results_all.csv"), index=False)

final_models = {
    "LightGBM_grouped_full_data": LGBMRegressor(**lgbm_group_params, random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
    "LightGBM_random_full_data": LGBMRegressor(**lgbm_random_params, random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
    "XGBoost_grouped_full_data": XGBRegressor(**xgb_group_params, random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
    "XGBoost_random_full_data": XGBRegressor(**xgb_random_params, random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
}

for name, model in final_models.items():
    model.fit(X, y)
    joblib.dump(model, os.path.join(OUTDIR, f"{name}.joblib"))

X_train_ad = group_results["X_train"]
X_test_ad = group_results["X_test"]

X_full_ad = np.vstack([X_train_ad, X_test_ad])

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
])
X_full_pca = pca_pipeline.fit_transform(X_full_ad)
n_components = pca_pipeline.named_steps["pca"].n_components_

k = 5
nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
nn.fit(X_full_pca)

ad_artifact = {
    "pca_pipeline": pca_pipeline,
    "nn": nn,
    "k": k,
}
joblib.dump(ad_artifact, os.path.join(OUTDIR, "applicability_domain_knn_pca.joblib"))

importance_group_lgbm, shap_values_group_lgbm, X_sample_group_lgbm = shap_and_importance_for_model(
    estimator=tuned_group_lgbm["estimator"],
    X_train=group_results["X_train"],
    feature_cols=feature_cols,
    out_prefix="grouped_lightgbm",
)
importance_random_lgbm, shap_values_random_lgbm, X_sample_random_lgbm = shap_and_importance_for_model(
    estimator=tuned_random_lgbm["estimator"],
    X_train=random_results["X_train"],
    feature_cols=feature_cols,
    out_prefix="random_lightgbm",
)
importance_group_xgb, shap_values_group_xgb, X_sample_group_xgb = shap_and_importance_for_model(
    estimator=tuned_group_xgb["estimator"],
    X_train=group_results["X_train"],
    feature_cols=feature_cols,
    out_prefix="grouped_xgboost",
)
importance_random_xgb, shap_values_random_xgb, X_sample_random_xgb = shap_and_importance_for_model(
    estimator=tuned_random_xgb["estimator"],
    X_train=random_results["X_train"],
    feature_cols=feature_cols,
    out_prefix="random_xgboost",
)
