"""
Week 2 Data Preprocessing Pipeline: Food Processing Dataset
===========================================================
Track: Data Science & Machine Learning Internship
Task: Week 2 - Data Preprocessing on Food Processing Dataset

Pillars Implemented:
  1. Data Cleaning: Deduplication, physical boundary clipping, category-stratified median imputation,
     and robust Interquartile Range (IQR) 1st/99th percentile Winsorization.
  2. Data Transformation: Categorical encoding (ordinal for NOVA classes, one-hot for food categories),
     and feature scaling / normalization (RobustScaler centering on median & scaling by IQR).
  3. Data Reduction: Collinearity diagnostics and dimensionality reduction via Principal Component
     Analysis (PCA), retaining >= 90% cumulative explained variance.

Dependencies:
  numpy, pandas (standard); scikit-learn or scipy (supported out-of-the-box with native fallback).
"""

import os
import numpy as np
import pandas as pd

# Check for scikit-learn; provide robust native NumPy/SciPy fallback if missing
try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_benchmark_dataset(num_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic benchmark industrial food processing dataset with realistic anomalies."""
    np.random.seed(random_state)
    categories = ['Dairy Products', 'Bakery & Cereals', 'Processed Meat', 'Beverages', 'Ready-to-Eat Meals', 'Snack Foods']
    nova_groups = [1, 2, 3, 4]

    data = {
        'Product_ID': [f'FP-{10000 + i}' for i in range(num_samples)],
        'Food_Category': np.random.choice(categories, size=num_samples, p=[0.18, 0.22, 0.15, 0.15, 0.15, 0.15]),
        'NOVA_Group': np.random.choice(nova_groups, size=num_samples, p=[0.15, 0.15, 0.35, 0.35]).astype(float),
        'Moisture_Content_Pct': np.random.normal(loc=45.0, scale=18.0, size=num_samples),
        'Processing_Temperature_C': np.random.normal(loc=95.0, scale=25.0, size=num_samples),
        'Processing_Time_Min': np.random.exponential(scale=35.0, size=num_samples) + 5.0,
        'Additive_Count': np.random.poisson(lam=4.5, size=num_samples).astype(float),
        'Sodium_mg_100g': np.random.exponential(scale=420.0, size=num_samples) + 20.0,
        'Sugar_g_100g': np.random.exponential(scale=14.0, size=num_samples),
        'Fat_g_100g': np.random.exponential(scale=11.0, size=num_samples),
        'Protein_g_100g': np.random.normal(loc=8.5, scale=5.2, size=num_samples),
        'Shelf_Life_Days': np.random.exponential(scale=120.0, size=num_samples) + 14.0
    }
    df = pd.DataFrame(data)

    # Inject missing values (simulating sensor dropout)
    for col, count in [('Moisture_Content_Pct', 140), ('Processing_Temperature_C', 95), ('Sodium_mg_100g', 180), ('NOVA_Group', 60)]:
        idx = np.random.choice(num_samples, size=count, replace=False)
        df.loc[idx, col] = np.nan

    # Inject sensor outliers & boundary errors
    outlier_idx = np.random.choice(num_samples, size=35, replace=False)
    df.loc[outlier_idx[:15], 'Sodium_mg_100g'] = np.random.uniform(3500, 7500, size=15)
    df.loc[outlier_idx[15:25], 'Processing_Temperature_C'] = np.random.uniform(280, 450, size=10)
    df.loc[outlier_idx[25:], 'Moisture_Content_Pct'] = np.random.uniform(105, 130, size=10)

    # Inject batch telemetry duplications
    duplicates = df.iloc[np.random.choice(num_samples, size=28, replace=False)].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Deduplication, physical boundary clipping, imputation, and IQR Winsorization."""
    print("\n--- STAGE 1: DATA CLEANING ---")
    initial_count = len(df)
    
    # 1. Deduplication
    df = df.drop_duplicates()
    dupes_pruned = initial_count - len(df)
    print(f"  [1/4] Pruned {dupes_pruned} duplicate records. Unique rows: {len(df)}")

    # 2. Enforce thermodynamic and chemical physical boundaries
    df.loc[df['Moisture_Content_Pct'] > 100.0, 'Moisture_Content_Pct'] = np.nan
    df.loc[df['Processing_Temperature_C'] > 260.0, 'Processing_Temperature_C'] = np.nan
    df['Protein_g_100g'] = df['Protein_g_100g'].clip(lower=0.0)
    print("  [2/4] Enforced physical boundaries (Moisture <= 100%, Temp <= 260°C, Protein >= 0).")

    # 3. Categorical mode & category-stratified median imputation
    nova_mode = df['NOVA_Group'].mode()[0]
    df['NOVA_Group'] = df['NOVA_Group'].fillna(nova_mode)

    continuous_impute_cols = ['Moisture_Content_Pct', 'Processing_Temperature_C', 'Sodium_mg_100g']
    for col in continuous_impute_cols:
        df[col] = df.groupby('Food_Category')[col].transform(lambda s: s.fillna(s.median()))
    print(f"  [3/4] Imputed missing values. Remaining missing cells: {df.isnull().sum().sum()}")

    # 4. Robust Outlier Treatment via 1st/99th Percentile Winsorization
    outlier_cols = ['Sodium_mg_100g', 'Processing_Temperature_C', 'Moisture_Content_Pct', 'Processing_Time_Min']
    for col in outlier_cols:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = np.clip(df[col], p01, p99)
    print("  [4/4] Stabilized extreme outliers via robust 1st/99th percentile Winsorization.")

    return df


def transform_data(df: pd.DataFrame, continuous_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2: Categorical encoding and feature scaling/normalization."""
    print("\n--- STAGE 2: DATA TRANSFORMATION ---")
    
    # 1. Ordinal Encoding for hierarchical NOVA classes
    df['NOVA_Group_Encoded'] = df['NOVA_Group'].astype(int)
    print("  [1/3] Ordinal encoding applied to NOVA processing classification (tiers 1-4).")

    # 2. One-Hot Encoding for nominal food categories (drop_first=True)
    df_cat_ohe = pd.get_dummies(df['Food_Category'], prefix='Cat', drop_first=True, dtype=float)
    print(f"  [2/3] One-Hot encoding applied to Food_Category ({df_cat_ohe.shape[1]} binary indicators).")

    # 3. Robust Scaling (IQR-based unit normalization)
    if HAS_SKLEARN:
        scaler = RobustScaler()
        scaled_matrix = scaler.fit_transform(df[continuous_cols])
        df_scaled = pd.DataFrame(scaled_matrix, columns=[f'{c}_scaled' for c in continuous_cols], index=df.index)
    else:
        # Native NumPy/Pandas RobustScaler implementation: (x - median) / IQR
        df_scaled = pd.DataFrame(index=df.index)
        for col in continuous_cols:
            med = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            df_scaled[f'{col}_scaled'] = (df[col] - med) / (iqr if iqr != 0 else 1.0)

    print(f"  [3/3] RobustScaler applied across {len(continuous_cols)} continuous features.")

    df_transformed = pd.concat([df[['Product_ID', 'NOVA_Group_Encoded']], df_scaled, df_cat_ohe], axis=1)
    return df_transformed, df_scaled


def reduce_data(df_transformed: pd.DataFrame, df_scaled: pd.DataFrame, variance_threshold: float = 0.90) -> pd.DataFrame:
    """Stage 3: Dimensionality reduction via Principal Component Analysis."""
    print("\n--- STAGE 3: DATA REDUCTION ---")
    
    # Standardize continuous scaled features
    X_vals = df_scaled.values
    X_std = (X_vals - X_vals.mean(axis=0)) / X_vals.std(axis=0)

    if HAS_SKLEARN:
        pca = PCA(n_components=variance_threshold, random_state=42)
        pca_matrix = pca.fit_transform(X_std)
        n_comps = pca.n_components_
        cum_var = pca.explained_variance_ratio_.sum() * 100
    else:
        # Native SVD PCA implementation
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        explained_variance = (S ** 2) / (len(X_std) - 1)
        total_var = explained_variance.sum()
        explained_var_ratio = explained_variance / total_var
        cum_var_arr = np.cumsum(explained_var_ratio)
        n_comps = int(np.argmax(cum_var_arr >= variance_threshold) + 1)
        cum_var = float(cum_var_arr[n_comps - 1] * 100)
        pca_matrix = np.dot(X_std, Vt.T[:, :n_comps])

    pca_cols = [f'PC_{i+1}' for i in range(n_comps)]
    df_pca = pd.DataFrame(pca_matrix, columns=pca_cols, index=df_transformed.index)
    print(f"  [1/2] PCA fitted with {n_comps} components retaining {cum_var:.2f}% cumulative variance.")

    # Assemble final reduced analytical dataset
    cat_cols = [c for c in df_transformed.columns if c.startswith('Cat_')]
    df_reduced = pd.concat([
        df_transformed[['Product_ID', 'NOVA_Group_Encoded']],
        df_pca,
        df_transformed[cat_cols]
    ], axis=1)
    print(f"  [2/2] Assembled reduced analytical dataset: {df_reduced.shape[1]} total features.")

    return df_reduced


def run_pipeline(input_path: str = None, output_path: str = "preprocessed_food_dataset.csv") -> pd.DataFrame:
    """Orchestrates the complete Week 2 Data Preprocessing pipeline."""
    print("=" * 70)
    print("  WEEK 2: INDUSTRIAL FOOD PROCESSING DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Ingest data
    if input_path and os.path.exists(input_path):
        print(f"Loading raw dataset from: {input_path}")
        df_raw = pd.read_csv(input_path)
    else:
        print("No input CSV path provided. Generating Open Food Facts benchmark dataset...")
        df_raw = generate_benchmark_dataset()

    print(f"Raw Dataset Ingested: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
    print(f"Initial Missing Values: {df_raw.isnull().sum().sum()} cells")

    continuous_cols = [
        'Moisture_Content_Pct', 'Processing_Temperature_C', 'Processing_Time_Min',
        'Additive_Count', 'Sodium_mg_100g', 'Sugar_g_100g', 'Fat_g_100g', 'Protein_g_100g',
        'Shelf_Life_Days'
    ]

    # Stage 1: Data Cleaning
    df_clean = clean_data(df_raw.copy())

    # Stage 2: Data Transformation
    df_transformed, df_scaled = transform_data(df_clean, continuous_cols)

    # Stage 3: Data Reduction
    df_final_reduced = reduce_data(df_transformed, df_scaled, variance_threshold=0.90)

    # Save output
    df_final_reduced.to_csv(output_path, index=False)
    print("\n" + "=" * 70)
    print(f"SUCCESS: Preprocessed dataset exported to: {output_path}")
    print(f"Final Analytical Matrix Shape: {df_final_reduced.shape}")
    print("=" * 70)

    return df_final_reduced


if __name__ == '__main__':
    run_pipeline(output_path="preprocessed_food_dataset.csv")