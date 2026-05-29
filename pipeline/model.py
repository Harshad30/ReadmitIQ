import duckdb # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import json

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "readmitiq.duckdb"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Features we'll use for prediction ─────────────────────────────────────
CATEGORICAL_FEATURES = [
    "DGNS_CATEGORY",      # type of illness
    "ADMISSION_TYPE",     # emergency vs elective etc
    "DISCHARGE_STATUS",   # where patient went after discharge
    "AGE_GROUP",          # age bucket
    "SEX",                # sex
    "RACE",               # race
    "LOS_GROUP",          # length of stay bucket
    "PRVDR_STATE_CD",     # state
]

NUMERIC_FEATURES = [
    "CLM_TOT_CHRG_AMT",   # total charges
    "CLM_PMT_AMT",        # medicare payment
    "CLM_UTLZTN_DAY_CNT", # length of stay in days
    "COST_COVERAGE_RATIO", # medicare coverage ratio
]

TARGET = "READMITTED_30D"


def load_features(con) -> pd.DataFrame:
    df = con.execute("SELECT * FROM claims_features").df()
    print(f"Loaded {len(df):,} rows from claims_features")
    return df


def prepare_ml_dataset(df: pd.DataFrame) -> tuple:
    print("Preparing ML dataset...")

    # select only the columns we need
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    df_ml = df[feature_cols + [TARGET]].copy()

    # drop rows with missing values in features
    before = len(df_ml)
    df_ml = df_ml.dropna()
    print(f"  Dropped {before - len(df_ml):,} rows with missing features")
    print(f"  Final dataset: {len(df_ml):,} rows")

    # encode categoricals as integers for XGBoost
    # LabelEncoder converts "Emergency" → 0, "Elective" → 1 etc
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    X = df_ml[feature_cols]
    y = df_ml[TARGET]

    print(f"  Class balance — Readmitted: {y.mean()*100:.1f}% | Not readmitted: {(1-y.mean())*100:.1f}%")
    return X, y, encoders


def train_baseline(X_train, y_train, X_test, y_test):
    """Logistic regression as baseline — always compare ML to a simpler model"""
    print("\nTraining baseline (Logistic Regression)...")
    lr = LogisticRegression(max_iter=3000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"  Baseline AUC-ROC: {auc:.3f}")
    print(f"  Baseline Avg Precision: {ap:.3f}")
    return lr, auc


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost...")

    # scale_pos_weight handles class imbalance
    # if 30% readmitted, weight = 70/30 = 2.33
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"  Class weight scale: {scale:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,   # handles imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    print(f"  XGBoost AUC-ROC: {auc:.3f}")
    print(f"  XGBoost Avg Precision: {ap:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"]))

    return model, auc


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 10 most predictive features:")
    print(importance.head(10).to_string(index=False))
    return importance


def cross_validate(model, X, y):
    print("\nRunning 5-fold cross validation...")

    # separate model without early stopping for CV
    # early stopping needs an eval set which CV doesn't provide
    cv_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
    print(f"  CV AUC scores: {[round(s, 3) for s in scores]}")
    print(f"  Mean AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
    return scores


def save_results(model, importance, cv_scores, baseline_auc, xgb_auc):
    # save model
    model.save_model(str(MODELS_DIR / "xgboost_readmission.json"))

    # save feature importance
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    # save metrics summary
    metrics = {
        "baseline_auc": round(baseline_auc, 3),
        "xgboost_auc": round(xgb_auc, 3),
        "cv_mean_auc": round(cv_scores.mean(), 3),
        "cv_std": round(cv_scores.std(), 3),
        "improvement_over_baseline": round(xgb_auc - baseline_auc, 3),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to models/xgboost_readmission.json")
    print(f"Metrics saved to models/metrics.json")


def run():
    con = duckdb.connect(str(DB_PATH))
    df = load_features(con)
    con.close()

    X, y, encoders = prepare_ml_dataset(df)

    # stratified split preserves class ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # train both models
    lr_model, baseline_auc = train_baseline(X_train, y_train, X_test, y_test)
    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)

    # feature importance
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    importance = get_feature_importance(xgb_model, feature_cols)

    # cross validation
    cv_scores = cross_validate(xgb_model, X, y)

    # save everything
    save_results(xgb_model, importance, cv_scores, baseline_auc, xgb_auc)

    print("\nModel training complete.")


if __name__ == "__main__":
    run()