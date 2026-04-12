import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import joblib


def load_adult_dataframe():
    """
    Adult Census Income (OpenML, version 2). ~49k rows after deduplication.
    Binary target: 1 if income >50K (~24% positive).
    First run downloads data (cached under scikit_learn_data).
    """
    print("Fetching Adult Census Income from OpenML (cached after first download)...")
    raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = raw.frame.copy()
    target_col = "class"
    if target_col not in df.columns:
        raise ValueError("Adult frame missing 'class' column")

    lab = df[target_col].astype(str).str.strip()
    df[target_col] = (lab == ">50K").astype(np.int64)
    df.dropna(subset=[target_col], inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows.")
    print(f"Target is {target_col} (1 => income >50K)")
    return df, target_col


def build_preprocessing_pipeline(df, target_col):
    """
    Identifies numerical and categorical cols.
    Returns two transformers: one for ordinal encoding, one for one-hot encoding.
    """
    features = [c for c in df.columns if c != target_col]
    num_cols = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[features].select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline_ord = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    cat_pipeline_ohe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor_ord = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline_ord, cat_cols),
        ],
        remainder="passthrough",
    )

    preprocessor_ohe = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline_ohe, cat_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor_ord, preprocessor_ohe, num_cols, cat_cols


def preprocess_and_save():
    df, target_col = load_adult_dataframe()

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print(f"Rows={len(y)}, target distribution: {np.bincount(y)}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )

    print(f"Train/Val/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    preprocessor_ord, preprocessor_ohe, num_cols, cat_cols = build_preprocessing_pipeline(
        X_train, target_col
    )

    X_train_ord = preprocessor_ord.fit_transform(X_train)
    X_val_ord = preprocessor_ord.transform(X_val)
    X_test_ord = preprocessor_ord.transform(X_test)

    X_train_ohe = preprocessor_ohe.fit_transform(X_train)
    X_val_ohe = preprocessor_ohe.transform(X_val)
    X_test_ohe = preprocessor_ohe.transform(X_test)

    os.makedirs("data/metadata", exist_ok=True)
    joblib.dump(preprocessor_ord, "data/metadata/preprocessor_ord.joblib")
    joblib.dump(preprocessor_ohe, "data/metadata/preprocessor_ohe.joblib")

    ohe_cat_cols = preprocessor_ohe.named_transformers_["cat"][
        "encoder"
    ].get_feature_names_out(cat_cols).tolist()
    ord_feature_names = num_cols + cat_cols
    ohe_feature_names = num_cols + ohe_cat_cols

    joblib.dump(
        {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "ord_features": ord_feature_names,
            "ohe_features": ohe_feature_names,
            "tabular_dataset": "adult",
            "target_col": target_col,
        },
        "data/metadata/feature_names.joblib",
    )

    with open("data/metadata/active_dataset.txt", "w", encoding="utf-8") as f:
        f.write("adult\n")

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        "data/processed/data_ord.npz",
        X_train=X_train_ord,
        y_train=y_train.values,
        X_val=X_val_ord,
        y_val=y_val.values,
        X_test=X_test_ord,
        y_test=y_test.values,
    )

    np.savez_compressed(
        "data/processed/data_ohe.npz",
        X_train=X_train_ohe,
        y_train=y_train.values,
        X_val=X_val_ohe,
        y_val=y_val.values,
        X_test=X_test_ohe,
        y_test=y_test.values,
    )

    X_train.to_csv("data/processed/X_train_raw.csv", index=False)
    X_val.to_csv("data/processed/X_val_raw.csv", index=False)
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Preprocessing successful.")


if __name__ == "__main__":
    preprocess_and_save()
