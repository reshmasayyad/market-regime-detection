"""Training-calibrated risk/trend rules and time-agnostic clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .features import MODEL_FEATURES


def rule_threshold(train):
    return float(train.volatility_20.quantile(.70))


def rule_labels(features, threshold):
    elevated = features.volatility_20.ge(threshold)
    rising = features.momentum_20.ge(0)
    labels = np.select([elevated & rising, elevated & ~rising, ~elevated & rising],
                       ["Volatile rising", "Volatile falling", "Calm rising"], default="Calm falling")
    return pd.Series(labels, index=features.index, name="rule_regime")


def fit_kmeans(train, validation, refit, counts=(3, 4)):
    scaler = StandardScaler().fit(train[MODEL_FEATURES])
    xtrain = scaler.transform(train[MODEL_FEATURES])
    xvalid = scaler.transform(validation[MODEL_FEATURES])
    rows = []
    for k in counts:
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(xtrain)
        labels = model.predict(xvalid)
        unique, sizes = np.unique(labels, return_counts=True)
        silhouette = float(silhouette_score(xvalid, labels, sample_size=min(1500, len(labels)), random_state=42)) if 1<len(unique)<len(labels) else np.nan
        rows.append({"k": k, "validation_silhouette": silhouette, "minimum_validation_share": sizes.min()/len(labels),
                     "validation_clusters_observed": len(unique)})
    selection = pd.DataFrame(rows)
    eligible = selection.loc[selection.minimum_validation_share.ge(.01) & selection.validation_clusters_observed.eq(selection.k)]
    if eligible.empty:
        raise RuntimeError("No K-means candidate has adequate validation cluster sizes")
    k = int(eligible.sort_values("validation_silhouette", ascending=False).iloc[0].k)
    final_scaler = StandardScaler().fit(refit[MODEL_FEATURES])
    model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(final_scaler.transform(refit[MODEL_FEATURES]))
    return {"model": model, "scaler": final_scaler, "selected_k": k, "selection": selection}


def kmeans_labels(bundle, features):
    return bundle["model"].predict(bundle["scaler"].transform(features[MODEL_FEATURES]))
