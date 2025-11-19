from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "churn"


def _separate_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target_col], errors="ignore")
    categorical_cols: List[str] = feature_df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric_cols: List[str] = feature_df.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()
    return categorical_cols, numeric_cols


def build_preprocessor(df: pd.DataFrame, target_col: str = TARGET_COL) -> ColumnTransformer:
    """Create a ColumnTransformer that one-hot encodes categoricals and scales numerics."""
    categorical_cols, numeric_cols = _separate_features(df, target_col)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor


def apply_preprocessor(preprocessor: ColumnTransformer, df: pd.DataFrame, target_col: str = TARGET_COL):
    """Fit_transform df without target column; return transformed features."""
    features = df.drop(columns=[target_col], errors="ignore")
    return preprocessor.transform(features)
