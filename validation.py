import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

VALIDATION_FILE = "datasetname_features.csv"
MODEL_DIR       = "Results_datasetname_MLM"
OUTDIR          = "Validation_v3"
RANDOM_STATE    = 42
N_BOOTSTRAP     = 1000
RMSE_LIMIT      = 20.0

os.makedirs(OUTDIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias}


def bootstrap_metrics(y_true, y_pred, n=N_BOOTSTRAP, ci=0.95, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    records = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        records.append(compute_metrics(y_true[s], y_pred[s]))
    df = pd.DataFrame(records)
    alpha = (1 - ci) / 2
    result = {}
    for col in df.columns:
        result[col] = {
            "mean": df[col].mean(),
            "lower": df[col].quantile(alpha),
            "upper": df[col].quantile(1 - alpha),
        }
    return result, df


def get_knn_distances(X_ext, pca_pipeline, nn):
    X_pca = pca_pipeline.transform(X_ext)
    dists, _ = nn.kneighbors(X_pca)
    return dists.mean(axis=1)


def cumulative_rmse_curve(sorted_y_true, sorted_y_pred, cutoff_idx):
    return np.array([
        np.sqrt(mean_squared_error(sorted_y_true[:i], sorted_y_pred[:i]))
        for i in cutoff_idx
    ])


def cumulative_r2_curve(sorted_y_true, sorted_y_pred, cutoff_idx):
    return np.array([
        r2_score(sorted_y_true[:i], sorted_y_pred[:i])
        for i in cutoff_idx
    ])


def ad_threshold_for_model(y_true, y_pred, knn_distances, rmse_limit):
    sorted_idx = np.argsort(knn_distances)
    sorted_dists = knn_distances[sorted_idx]
    sorted_y_true = y_true[sorted_idx]
    sorted_y_pred = y_pred[sorted_idx]
    n = len(sorted_y_true)

    percentiles = np.arange(0, 101)
    cutoff_idx = np.ceil(percentiles / 100 * n).astype(int).clip(1, n)
    cum_rmse = cumulative_rmse_curve(sorted_y_true, sorted_y_pred, cutoff_idx)
    cum_r2 = cumulative_r2_curve(sorted_y_true, sorted_y_pred, cutoff_idx)

    valid = np.where(cum_rmse <= rmse_limit)[0]
    if len(valid) > 0:
        threshold_pct = int(percentiles[valid[-1]])
        threshold_knn = float(sorted_dists[cutoff_idx[valid[-1]] - 1])
    else:
        threshold_pct = 0
        threshold_knn = float(sorted_dists.min() - 1e-12)

    in_domain_mask = knn_distances <= threshold_knn
    coverage = in_domain_mask.mean() * 100

    return {
        "threshold_pct": threshold_pct,
        "threshold_knn": threshold_knn,
        "coverage": coverage,
        "in_domain_mask": in_domain_mask,
        "cumulative_rmse": cum_rmse,
        "cumulative_r2": cum_r2,
        "sorted_idx": sorted_idx,
        "sorted_dists": sorted_dists,
        "percentiles": percentiles,
        "cutoff_idx": cutoff_idx,
    }


df_ext = pd.read_csv(VALIDATION_FILE)
y_ext = df_ext["Stokes Shift (nm)"].values

feat_df = pd.read_csv(os.path.join(MODEL_DIR, "feature_names.csv"))
feature_cols = feat_df["Feature"].tolist()

missing_cols = [c for c in feature_cols if c not in df_ext.columns]
for c in missing_cols:
    df_ext[c] = 0.0

X_ext_raw = df_ext[feature_cols].copy()
imputer = joblib.load(os.path.join(MODEL_DIR, "median_imputer.joblib"))
X_ext = pd.DataFrame(imputer.transform(X_ext_raw), columns=feature_cols)

model_files = {
    "LightGBM_grouped": "LightGBM_grouped_full_data.joblib",
    "LightGBM_random":  "LightGBM_random_full_data.joblib",
    "XGBoost_grouped":  "XGBoost_grouped_full_data.joblib",
    "XGBoost_random":   "XGBoost_random_full_data.joblib",
}

models = {}
for name, fname in model_files.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        models[name] = joblib.load(path)

ad_artifact = joblib.load(os.path.join(MODEL_DIR, "applicability_domain_knn_pca.joblib"))
pca_pipeline = ad_artifact["pca_pipeline"]
nn = ad_artifact["nn"]
ad_k = ad_artifact["k"]

knn_distances = get_knn_distances(X_ext, pca_pipeline, nn)
predictions = {name: model.predict(X_ext) for name, model in models.items()}

ad_results = {
    name: ad_threshold_for_model(y_ext, y_pred, knn_distances, RMSE_LIMIT)
    for name, y_pred in predictions.items()
}

metric_rows = []
for name, y_pred in predictions.items():
    in_domain_mask = ad_results[name]["in_domain_mask"]
    metric_rows.append({"Model": name, "Subset": "All", **compute_metrics(y_ext, y_pred)})
    if in_domain_mask.sum() > 0:
        metric_rows.append({"Model": name, "Subset": "In-domain",
                             **compute_metrics(y_ext[in_domain_mask], y_pred[in_domain_mask])})

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(os.path.join(OUTDIR, "external_metrics.csv"), index=False)

boot_rows = {}
for name, y_pred in predictions.items():
    ci_result, _ = bootstrap_metrics(y_ext, y_pred, ci=0.95)
    boot_rows[name] = ci_result

summary_rows = []
for name, y_pred in predictions.items():
    m = compute_metrics(y_ext, y_pred)
    boot95 = boot_rows[name]
    ad = ad_results[name]
    in_domain_mask = ad["in_domain_mask"]
    m_in = compute_metrics(y_ext[in_domain_mask], y_pred[in_domain_mask]) if in_domain_mask.sum() > 0 else {}

    summary_rows.append({
        "Model": name,
        "MAE (nm)": round(m["MAE"], 2),
        "RMSE (nm)": round(m["RMSE"], 2),
        "R2": round(m["R2"], 3),
        "Bias (nm)": round(m["Bias"], 2),
        "MAE 95% CI": f"[{boot95['MAE']['lower']:.1f}, {boot95['MAE']['upper']:.1f}]",
        "RMSE 95% CI": f"[{boot95['RMSE']['lower']:.1f}, {boot95['RMSE']['upper']:.1f}]",
        "R2 95% CI": f"[{boot95['R2']['lower']:.3f}, {boot95['R2']['upper']:.3f}]",
        "In-domain MAE (nm)": round(m_in.get("MAE", np.nan), 2),
        "In-domain RMSE (nm)": round(m_in.get("RMSE", np.nan), 2),
        "In-domain R2": round(m_in.get("R2", np.nan), 3),
        "AD threshold pct": ad["threshold_pct"],
        "AD threshold d": round(ad["threshold_knn"], 4),
        "AD coverage (%)": round(ad["coverage"], 1),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTDIR, "validation_summary.csv"), index=False)
