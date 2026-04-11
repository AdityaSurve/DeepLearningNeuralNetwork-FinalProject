import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(path='../../dataset/heart_2020_cleaned.csv'):
    # Adjust path if called from experiments/ or project root
    if not os.path.exists(path):
        path = 'dataset/heart_2020_cleaned.csv'
    return pd.read_csv(path)

def build_preprocessing_pipeline(df, target_col):
    """
    Identifies numerical and categorical cols.
    Returns two transformers: one for ordinal encoding, one for one-hot encoding.
    """
    features = [c for c in df.columns if c != target_col]
    num_cols = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline - Ordinal (for DL embeddings & trees)
    cat_pipeline_ord = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Categorical pipeline - OneHot (for LR, MLP, etc)
    cat_pipeline_ohe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor_ord = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline_ord, cat_cols)
    ], remainder='passthrough')
    
    preprocessor_ohe = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline_ohe, cat_cols)
    ], remainder='passthrough')
    
    return preprocessor_ord, preprocessor_ohe, num_cols, cat_cols

def preprocess_and_save():
    print("Loading data...")
    df = load_data()
    
    # Clean duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicates.")
    
    target_col = 'HeartDisease'
    if target_col not in df.columns:
        # Fallback target
        targets = [c for c in df.columns if 'heart' in c.lower() or 'michd' in c.lower()]
        target_col = targets[0]
        
    print(f"Target is {target_col}")
    
    # Map target strings to 1/0
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        # Drop missing targets
        df.dropna(subset=[target_col], inplace=True)
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    print(f"Target distribution: {np.bincount(y)}")
    
    # Splitting 70/15/15
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    ) # 0.1765 of 0.85 is ~0.15
    
    print(f"Train/Val/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    
    preprocessor_ord, preprocessor_ohe, num_cols, cat_cols = build_preprocessing_pipeline(X_train, target_col)
    
    # Fit and transform
    X_train_ord = preprocessor_ord.fit_transform(X_train)
    X_val_ord = preprocessor_ord.transform(X_val)
    X_test_ord = preprocessor_ord.transform(X_test)
    
    X_train_ohe = preprocessor_ohe.fit_transform(X_train)
    X_val_ohe = preprocessor_ohe.transform(X_val)
    X_test_ohe = preprocessor_ohe.transform(X_test)
    
    # Save transformers
    os.makedirs('data/metadata', exist_ok=True)
    joblib.dump(preprocessor_ord, 'data/metadata/preprocessor_ord.joblib')
    joblib.dump(preprocessor_ohe, 'data/metadata/preprocessor_ohe.joblib')
    
    # Save column names for reference
    ohe_cat_cols = preprocessor_ohe.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols).tolist()
    ord_feature_names = num_cols + cat_cols
    ohe_feature_names = num_cols + ohe_cat_cols
    
    joblib.dump({'num_cols': num_cols, 'cat_cols': cat_cols, 
                 'ord_features': ord_feature_names, 'ohe_features': ohe_feature_names}, 
                'data/metadata/feature_names.joblib')
    
    # Save to disk as npz for speed and type safety
    os.makedirs('data/processed', exist_ok=True)
    np.savez_compressed('data/processed/data_ord.npz', 
                        X_train=X_train_ord, y_train=y_train.values,
                        X_val=X_val_ord, y_val=y_val.values,
                        X_test=X_test_ord, y_test=y_test.values)
    
    np.savez_compressed('data/processed/data_ohe.npz', 
                        X_train=X_train_ohe, y_train=y_train.values,
                        X_val=X_val_ohe, y_val=y_val.values,
                        X_test=X_test_ohe, y_test=y_test.values)

    # Raw feature rows (same order as npz) for fairness / bias mitigation on train & val.
    X_train.to_csv("data/processed/X_train_raw.csv", index=False)
    X_val.to_csv("data/processed/X_val_raw.csv", index=False)
    # Test set for fairness evaluation scripts.
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("Preprocessing successful.")

if __name__ == '__main__':
    preprocess_and_save()
